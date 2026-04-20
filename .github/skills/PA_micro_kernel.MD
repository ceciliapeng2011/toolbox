## PagedAttention stage and kernel selection skills

### Stage decision (Intel GPU)

- `PagedAttentionStage::GENERATE` when `query_shape[0] == past_lens_shape[0]`.  【i.e. `batch_size_in_tokens == batch_size_in_sequences`, which means every sequence has 1 input token in query.】
There must be at least 1 item in query for each subsequence.
- `PagedAttentionStage::MIXED` when shapes differ and any `past_lens` value is non‑zero.
- `PagedAttentionStage::PREFILL` when shapes differ and all `past_lens` values are zero.

Reference: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp#L215-L241)

### Kernel selection (micro SDPA enabled)

- PREFILL → `sdpa_micro__prefill`
- MIXED → `sdpa_micro__generate` (mixed path)
- GENERATE → `sdpa_micro__gqa_single_token` only when GQA micro conditions are met; otherwise non‑micro single‑token kernels are used [i.e. `paged_attention_opt__single_token`].

#### GQA micro conditions (GENERATE stage):

- `kv_group_size == 8` and `k_head_size == 64`.
- When not met, micro SDPA is disabled for GENERATE.


### Kernel selection (micro SDPA disabled)

- PREFILL → `sdpa_opt__multi_tokens`
- MIXED → `paged_attention_opt__multi_tokens` (+ `paged_attention_opt__multi_tokens_finalization` when partitions > 1)
- GENERATE → `paged_attention_opt__single_token` or `paged_attention_opt__gqa_single_token` (+ finalization when partitions > 1)

Reference: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp#L1386-L1410)

## Micro PA v.s. CM PA kernel
- **sdpa_micro__prefill scheduling & workload splitting (summary)**:
	- Query tiles are sized by `wg_tile_q` (aka micro GEMM `wg_tile_n`). Each subsequence is split into blocks of this size; blocks never cross subsequence boundaries.
	- Total `n_queries` for dispatch is the sum of *aligned* per‑subsequence prompt lengths, where each prompt is aligned to `wg_tile_q` using `subsequence_begins`.
	- GWS is tiled over query blocks × heads: `gws[0] = lws[0] * ceil_div(n_queries, wg_tile_q)`, `gws[1] = lws[1] * heads`, `gws[2] = 1` for paged‑attention path.
	- A `(block_start, subsequence_idx)` mapping buffer assigns each query block to exactly one subsequence for execution.

- **sdpa_micro__generate (MIXED path) subsequence/location details**:
	- `sdpa_micro__generate` does not linearly scan subsequences in kernel. Instead, host prepares a packed mapping table in an internal buffer as pairs:
		- `[block_start_pos_0, subsequence_idx_0, block_start_pos_1, subsequence_idx_1, ...]`.
	- For each dispatched query block WG, kernel reads one pair using `query_block_idx = get_group_id(0) << 1`:
		- `block_start_pos = mapping[query_block_idx]`
		- `subsequence_idx = mapping[query_block_idx + 1]`
	- Kernel locates subsequence range via `subsequence_begins`:
		- `subsequence_begin = subsequence_begins[subsequence_idx]`
		- `subsequence_end = subsequence_begins[subsequence_idx + 1]`
	- Kernel locates query position inside that subsequence by subtraction:
		- `subsequence_query_block_idx = block_start_pos - subsequence_begin`.
	- In MIXED/generate-mode micro path, key length is derived per subsequence as:
		- `k = q + past_lens[subsequence_idx]`.
	- This is an O(1) WG→subsequence lookup design (descriptor-table style), replacing branchy per-WG scans.

- **Why this locating method is usually better**:
	- Pros:
		- Lower kernel control overhead and less branch divergence.
		- Better scaling when subsequence count is high or lengths are skewed.
		- Deterministic WG ownership (each query block maps to exactly one subsequence).
	- Tradeoffs:
		- Extra host-side preprocessing each step to build mapping pairs.
		- Additional mapping buffer traffic and tighter host↔kernel ABI coupling.
		- For very tiny workloads, mapping-build overhead can partially offset runtime gain.

- **Code references for sdpa_micro__generate locating path**:
	- Mapping buffer format and fill: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp#L1566-L1569), [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp#L1886-L1890)
	- Micro kernel argument binding: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp#L1293)
	- Kernel-side pair decode and q-position computation: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L179-L206)

- **OCL vs CM perf conclusion (xe2, run.sh basic")**:
	- The benchmark compares *different block sizes*: OCL perf uses `block_size=16`, CM perf uses `block_size=256`. This materially changes cache layout, block tables, and KV traffic, so the observed CM speedup is not an apples‑to‑apples kernel comparison.
	- OCL micro prefill uses `wg_tile_q=64` (quantized, head_size=128 on xe2), while CM multi‑token uses `wg_seq_len=256` (q_step=16, WG_SIZE=16), resulting in fewer, larger CM tiles and lower per‑tile overhead for long prefill.



## Kernel Comparision: `paged_attention_opt__single_token` (OCL) vs `pa_single_token` (CM)

When traces show OCL decode kernel as `paged_attention_opt__single_token...` and CM decode kernel as `pa_single_token_cm...`, CM commonly shows lower latency for the following implementation reasons.

### Side-by-side implementation differences

| Aspect | OCL (`paged_attention_opt__single_token`) | CM (`pa_single_token`) |
|---|---|---|
| Kernel structure | Generic SDPA stage-0 + stage-1 finalization | Dedicated CM single-token + dedicated CM finalization |
| Hot math | subgroup broadcast + `mad` loops | DPAS (`cm_dpas`) tiled matrix ops |
| Synchronization | Multiple SLM barriers in stage-0 softmax/reduction | More register-centric accumulation, less barrier-heavy hot path |
| Memory movement | Generic block reads + SLM staging (`slm_qk_vals`, max/sum buffers) | Xe2 2D block descriptor loads (`lsc::block_2d_desc`) + VNNI-friendly layout |
| Dispatch mapping | GWS scaled by `head_size * num_partitions * sg_scale` | GWS over `kv_heads * q_head_chunks_per_kv_head * partitions` (chunked Q-head mapping) |

### Exact hot-loop references

- OCL main kernel entry: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_attention_opt.cl](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_attention_opt.cl#L48)
- OCL SLM and barrier-heavy flow: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_attention_opt.cl](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_attention_opt.cl#L151-L423)
- OCL finalization stage entry: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_attention_opt.cl](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_attention_opt.cl#L669)
- OCL single-token dispatch shape: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp#L430-L431)

- CM single-token DPAS accumulator (`rS`) and inner loops: [openvino/src/plugins/intel_gpu/src/graph/impls/cm/pa_single_token.cm](openvino/src/plugins/intel_gpu/src/graph/impls/cm/pa_single_token.cm#L157-L359)
- CM Xe2-oriented 2D block reads: [openvino/src/plugins/intel_gpu/src/graph/impls/cm/pa_single_token.cm](openvino/src/plugins/intel_gpu/src/graph/impls/cm/pa_single_token.cm#L170-L172), [openvino/src/plugins/intel_gpu/src/graph/impls/cm/pa_single_token.cm](openvino/src/plugins/intel_gpu/src/graph/impls/cm/pa_single_token.cm#L431-L433)
- CM PV DPAS path: [openvino/src/plugins/intel_gpu/src/graph/impls/cm/pa_single_token.cm](openvino/src/plugins/intel_gpu/src/graph/impls/cm/pa_single_token.cm#L541-L605)
- CM single-token dispatch shape: [openvino/src/plugins/intel_gpu/src/graph/impls/cm/paged_attention_gen.cpp](openvino/src/plugins/intel_gpu/src/graph/impls/cm/paged_attention_gen.cpp#L426-L440)

### Why CM has advantage in this comparison

1. Higher compute density in hot loops via `cm_dpas`.
2. Lower synchronization overhead than OCL stage-0 SLM/barrier sequence.
3. Better-aligned K/V tile loads for Xe2 via `lsc::block_2d_desc`.
4. More specialized decode mapping (Q-head chunking per KV head) with less generic-path overhead.

### Caveat

This applies when OCL actually runs `paged_attention_opt__single_token` (non-micro path). If OCL decode switches to microkernel (`sdpa_micro__gqa_single_token`), comparison basis changes.

### Q/A — Is it possible that existing OCL path uses microkernel path (`sdpa_micro__generate`) for decoding stage?

- Short answer: **No, not for pure GENERATE decode stage**.
- In current logic:
	- `sdpa_micro__generate` is used by **MIXED** path.
	- **GENERATE** can use only `sdpa_micro__gqa_single_token`, and only when GQA micro conditions are met (`kv_group_size == 8 && k_head_size == 64`).
	- Otherwise GENERATE falls back to non-micro kernels such as `paged_attention_opt__single_token` / `paged_attention_opt__gqa_single_token`.

Reference: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp#L1344-L1354)


## TurboQuant + CM PagedAttention test flow

Reference test file: [openvino/src/plugins/intel_gpu/tests/unit/test_cases/paged_attention_gpu_test.cpp](openvino/src/plugins/intel_gpu/tests/unit/test_cases/paged_attention_gpu_test.cpp)

### What the test builds

- `PagedAttentionManager` generates per-subsequence:
	- `query_data`: only current `num_tokens`
	- `key_data` / `value_data`: full logical sequence of `past_len + num_tokens`
- Runtime PA inputs are then split as:
	- `query` = current tokens
	- `key` / `value` = current tokens only (`skip_past_len=true`)
	- `key_cache` / `value_cache` = past tokens materialized into paged KV cache buffers

### How TurboQuant cache is prepared

- For TurboQuant key cache, the test uses CM-style cache packing via `get_key_cache_memory_cm()`.
- Past K tokens are normalized per token, rotated/quantized into 4-bit indices, packed into u8 payload, and stored with one fp16 norm per token.
- TurboQuant value cache is stored in compressed per-token form too, but value path is simpler: u8 payload + fp16 scale/zp tails.
- In 2nd-token / decode-style cases, the cache therefore represents the `past_len` portion, while fresh K/V still come from the non-cache `key` / `value` inputs.

### How expected output is computed

- `PagedAttentionReference::get_reference()` builds expected outputs per subsequence.
- If KV cache compression is enabled, it does **not** use original past K/V directly:
	- `read_key_from_cache()` reconstructs fp16 key vectors from the compressed cache layout.
	- `read_value_from_cache()` reconstructs fp16 value vectors from the compressed cache layout.
- The reconstructed fp16 K/V are then fed into a small reference attention graph (`gemm(Q,K)` → scale/mask → `softmax` → `gemm(...,V)`).
- Therefore, TurboQuant tests compare GPU PA against a reference that consumes **dequantized cache contents**, not untouched original past K/V.

### What is actually validated

- The PA primitive is executed with:
	- current-query input,
	- current K/V input,
	- compressed past KV cache,
	- usual paging metadata (`past_lens`, `block_indices`, `subsequence_begins`, etc.).
- Output is compared against reference with relaxed tolerance for compression:
	- compressed KV cache: higher tolerance
	- TurboQuant: slightly higher again (`30e-3`)

### TurboQuant-specific test cases

- `turboquant_test` is the positive suite.
- It is CM-only and guarded by `check_cm_available()` (CM JIT support + Xe2/Xe3).
- It enforces these assumptions before running:
	- exactly one subsequence
	- `block_size == 16`
	- KV cache compression enabled
	- `key_cache_quant_mode == TURBOQUANT`
	- no XAttention

Instantiated cases in `smoke_cm_turboquant`:

- decode/generate-style: `{{1, 64}}`
	- 1 fresh token, 64 past tokens
- prefill-style: `{{64, 0}}`
	- 64 fresh tokens, no past cache history

### Negative coverage

- `turboquant_invalid_test` checks TurboQuant + XAttention is rejected.
- Instantiated in `smoke_cm_turboquant_invalid_xattention`.
- Expected failure message contains: `TurboQuant is not supported with XAttention`.

### Practical interpretation

- The test is not only validating cache-update packing.
- It validates the end-to-end CM TurboQuant PA path:
	1. generate source Q/K/V
	2. pack past K/V into TurboQuant cache layout
	3. run CM paged attention with fresh K/V + compressed cache
	4. decode cache in reference path and compare final attention output

## unit test
- OV_VERBOSE=4 OV_GPU_TQ_PREFILL_EXTRA_PATH=1 ./bin/intel64/Debug/ov_gpu_unit_tests --gtest_filter="smoke_cm_turboquant/turboquant_test.basic/3" --gtest_output=xml:/tmp/tq_metrics.xml


## `sdpa_micro__prefill` — tile geometry, dispatch, register & occupancy analysis

### `sdpa_config_t` struct

Defined at [sdpa_gen_micro.cpp:253-258](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp#L253-L258):

```cpp
struct sdpa_config_t {
    int unroll_m_kq, unroll_n_kq;  // SG tile sizes for K*Q GEMM
    int unroll_m_vs, unroll_n_vs;  // SG tile sizes for V*S GEMM
    int wg_m_kq, wg_n_kq;         // # SGs tiling M,N for K*Q GEMM
    int wg_m_vs, wg_n_vs;         // # SGs tiling M,N for V*S GEMM
};
```

Derived sizes:
- `wg_tile = unroll × wg` (per axis)
- `sg_per_wg = wg_m × wg_n`
- `wg_tile_q` = `wg_tile_n` of KQ GEMM (drives dispatch granularity)
- Q tokens per work-item = `sg_tile_n / SUBGROUP_SIZE`

### Config selection (PA prefill, dynamic seq)

For PA prefill the sequence length is dynamic (`nkeys_v=0`), so the `seq <= 0 && is_pa` → `is_prefill ? &<prefill_config>` path is taken.

| Architecture | h128 config | h256 config |
|---|---|---|
| XeHPG | `xehpg_h128` = `{16,16,32,8, 8,2,4,4}` | `xehpg_h256` = `{16,16,32,8, 16,2,8,4}` |
| XeHPC | `xehpc_h128` = `{16,64,32,16, 16,2,4,8}` | `xehpc_h256` = `{16,32,32,32, 8,4,8,4}` |
| Xe2 | falls to `xehpc_h128` | falls to `xehpc_h256` |

References:
- XeHPG: [sdpa_gen_micro.cpp:480-504](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp#L480-L504) (h128), [sdpa_gen_micro.cpp:505-531](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp#L505-L531) (h256)
- XeHPC: [sdpa_gen_micro.cpp:600-633](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp#L600-L633) (h128), [sdpa_gen_micro.cpp:634-641](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp#L634-L641) (h256)
- Xe2: [sdpa_gen_micro.cpp:748-749](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp#L748-L749) (h128 → xehpc), [sdpa_gen_micro.cpp:782](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp#L782) (h256 → xehpc)

### Tile geometry comparison (Xe2/XeHPC)

#### h128 (`xehpc_h128 = {16, 64, 32, 16, 16, 2, 4, 8}`), D_MAX=128

| | KQ GEMM | VS GEMM |
|---|---|---|
| sg_tile_m × sg_tile_n | 16 × 64 | 32 × 16 |
| sg_per_wg (m × n) | 16 × 2 = 32 | 4 × 8 = 32 |
| **wg_tile_m** | 16×16 = **256** | 32×4 = **128** |
| **wg_tile_n (wg_tile_q)** | 64×2 = **128** | 16×8 = **128** |

#### h256 (`xehpc_h256 = {16, 32, 32, 32, 8, 4, 8, 4}`), D_MAX=256

| | KQ GEMM | VS GEMM |
|---|---|---|
| sg_tile_m × sg_tile_n | 16 × 32 | 32 × 32 |
| sg_per_wg (m × n) | 8 × 4 = 32 | 8 × 4 = 32 |
| **wg_tile_m** | 16×8 = **128** | 32×8 = **256** |
| **wg_tile_n (wg_tile_q)** | 32×4 = **128** | 32×4 = **128** |

**Key: `wg_tile_q` = 128 for both head sizes.** Same dispatch granularity, different internal factorization (h128: fewer wider SG columns; h256: more narrower SG columns).

### GEMM dimension mapping

#### KQ GEMM: `S[k_tokens × q_tokens] = K^T[k_tokens × d] × Q[d × q_tokens]`

| GEMM axis | Semantic | h128 per-WG | h256 per-WG |
|---|---|---|---|
| M | K tokens (iterated) | 256 | 128 |
| N | Q tokens (output cols) | 128 | 128 |
| K (inner) | head_size (dot product) | 128 | 256 |

#### VS GEMM: `A[d × q_tokens] = V[d × k_chunk] × softmax(S)[k_chunk × q_tokens]`

| GEMM axis | Semantic | h128 per-WG | h256 per-WG |
|---|---|---|---|
| M | head_size (output rows) | 128 | 256 |
| N | Q tokens | 128 | 128 |
| K (inner) | k_chunk (= KQ wg_tile_m) | 256 | 128 |

### Subgroup grid layouts

Each WG = 32 SGs. `sg_ij` (0..31) is mapped to a 2D grid per [sdpa_micro.cl:254-258](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L254-L258):

```c
uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;
uint sg_i_vs = sg_ij % ugemm_vs_sg_per_wg_m;
uint sg_j_vs = sg_ij / ugemm_vs_sg_per_wg_m;
```

#### h128 KQ: 16 rows × 2 cols (tall, narrow)

```
             ← 128 queries (N) →
              j=0          j=1
         ┌───────────┬───────────┐
  i=0    │  16 × 64  │  16 × 64  │  ↑
  i=1    │  16 × 64  │  16 × 64  │  │
  ...    │    ...     │    ...     │  256 keys (M)
  i=15   │  16 × 64  │  16 × 64  │  ↓
         └───────────┴───────────┘
```

#### h128 VS: 4 rows × 8 cols (short, wide)

```
             ← 128 queries (N) →
          j=0  j=1  j=2  j=3  j=4  j=5  j=6  j=7
         ┌────┬────┬────┬────┬────┬────┬────┬────┐
  i=0    │32×16│   │    │    │    │    │    │    │  ↑
  i=1    │    │    │    │    │    │    │    │    │  128 head dims (M)
  i=2    │    │    │    │    │    │    │    │    │  │
  i=3    │    │    │    │    │    │    │    │    │  ↓
         └────┴────┴────┴────┴────┴────┴────┴────┘
```

#### h256 KQ: 8 rows × 4 cols (square)

```
             ← 128 queries (N) →
          j=0    j=1    j=2    j=3
         ┌──────┬──────┬──────┬──────┐
  i=0    │16×32 │16×32 │16×32 │16×32 │  ↑
  ...    │      │      │      │      │  128 keys (M)
  i=7    │16×32 │16×32 │16×32 │16×32 │  ↓
         └──────┴──────┴──────┴──────┘
```

#### h256 VS: 8 rows × 4 cols (square)

```
             ← 128 queries (N) →
          j=0    j=1    j=2    j=3
         ┌──────┬──────┬──────┬──────┐
  i=0    │32×32 │32×32 │32×32 │32×32 │  ↑
  ...    │      │      │      │      │  256 head dims (M)
  i=7    │32×32 │32×32 │32×32 │32×32 │  ↓
         └──────┴──────┴──────┴──────┘
```

h128 flips between tall-narrow (KQ) and short-wide (VS). h256 keeps both GEMMs square. SG-to-position remapping between phases is free — S is communicated via SLM.

### Q tokens per work-item

Each work-item = 1 SIMD lane (SUBGROUP_SIZE=16). Q tokens per lane = `sg_tile_n / 16`.

| | KQ phase | VS phase |
|---|---|---|
| **h128** | 64/16 = **4** | 16/16 = **1** |
| **h256** | 32/16 = **2** | 32/16 = **2** |

h128 work-items handle 4 q-tokens during KQ but only 1 during VS (grid reshapes). h256 is symmetric at 2 q-tokens throughout.

### Dispatch (identical for both head sizes)

Reference: [sdpa_gen_micro.cpp:1330-1393](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp#L1330-L1393) and [sdpa_micro.cl:179-196](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L179-L196).

```
lws = {SUBGROUP_SIZE=16, sg_per_wg=32, 1} = {16, 32, 1}  → 512 work-items/WG
gws[0] = 16 × ceil_div(n_queries_aligned, 128)   ← 1 WG per 128-token query block
gws[1] = 32 × heads_num                           ← 1 WG per head
gws[2] = 1
```

Per-subsequence prompt lengths are aligned to 128 via `subsequence_begins`. Host-side mapping buffer (`blocked_indexes_start_and_gws_mapping`) maps each WG to `(block_start_pos, subsequence_idx)` — O(1) lookup.

### Register usage per work-item (one SIMD lane)

Tile register sizes are derived from `DECLARE_2D_TILE` at [tile_ops.cl:480](openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/include/batch_headers/tile_ops.cl#L480):
per-lane elements = `(br × bc / SUBGROUP_SIZE) × nbr × nbc`.

#### h256

| Tile | Per-lane elements | Type | GRFs |
|---|---|---|---|
| Q_tile (transient) | 128/16 × 4 = 32 | uint32 | 32 |
| S_tile (KQ result) | 16×32/16 = 32 | f32 | 32 |
| A_tile (VS accum) | 32×32/16 = 64 | f32 | **64** |
| A_tile1 (VS result) | 64 | f32 | **64** |
| S_sum/max/max_old | 2+2+2 | f32 | 6 |
| S_tile_half2 | 16 | uint32 | 16 |

#### h128

| Tile | Per-lane elements | Type | GRFs |
|---|---|---|---|
| Q_tile (transient) | 64/16 × 4 = 16 | uint32 | 16 |
| S_tile (KQ result) | 16×64/16 = **64** | f32 | **64** |
| A_tile (VS accum) | 32×16/16 = 32 | f32 | 32 |
| A_tile1 (VS result) | 32 | f32 | 32 |
| S_sum/max/max_old | 4+4+4 | f32 | 12 |
| S_tile_half2 | 32 | uint32 | 32 |

h128 pressure peaks during KQ (large S_tile=64 GRFs). h256 pressure peaks during VS (large A_tile+A_tile1=128 GRFs).

#### Peak register pressure by phase (estimated)

| Phase | h128 GRFs | h256 GRFs |
|---|---|---|
| KQ GEMM (S building + μkernel internals) | ~110-130 | ~90-110 |
| Softmax + S→SLM | ~112 | ~100 |
| VS GEMM (A_tile + A_tile1 + μkernel) | ~100-120 | ~160-180 |
| **Peak** | **~130** | **~180** |

Both exceed 128 GRFs → **256-GRF mode forced** ([sdpa_gen_micro.cpp:907-911](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_micro.cpp#L907-L911)).

### SLM footprint

From [sdpa_micro.cl:261-266](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L261-L266):

| Region | Formula | h128 | h256 |
|---|---|---|---|
| Q_slm | `D_MAX × wg_tile_n × 2` | 32 KB | 64 KB |
| S_slm | `wg_tile_m_kq × wg_tile_n × 2` | 64 KB | 32 KB |
| S_sum_slm | `wg_tile_n × sg_per_wg_m_kq × 4` | 8 KB | 4 KB |
| S_max_slm | `wg_tile_n × 4` | 512 B | 512 B |
| Total (excl μkernel SLM) | | **~104 KB** | **~100 KB** |
| Xe2 SLM per DSS | | 128 KB | 128 KB |
| **WGs per DSS** | | **1** | **1** |

Q_slm halves for h128 (smaller D_MAX) but S_slm doubles (bigger wg_tile_m). Both saturate SLM.

### Occupancy analysis (Xe2)

Xe2 Xe-core (1 DSS) = 8 EUs. Xe2 supports 8 threads/EU in 128-GRF, 4 threads/EU in 256-GRF.

| Factor | h128 | h256 |
|---|---|---|
| 256-GRF mode | Yes | Yes |
| Threads/EU (256-GRF) | 4 | 4 |
| WG size | 32 SGs = 32 threads | 32 threads |
| Threads/EU (32 threads / 8 EUs) | 4 | 4 |
| WGs per DSS (SLM-limited) | 1 | 1 |
| **Occupancy vs 256-GRF max** | **100%** | **100%** |
| **Occupancy vs full HW max (8 thr/EU)** | **50%** | **50%** |

Identical occupancy situation for both head sizes.

### Why ~50% of DPAS roofline is observed (and it's not just occupancy)

Reduced occupancy (4 vs 8 threads/EU) does **not** halve DPAS throughput — DPAS peak is the same regardless of thread count. More threads help hide **memory latency**, but for compute-bound DPAS chains this matters less. The real causes:

**1. Non-DPAS phases consume a large fraction of each iteration**

Per k-block iteration structure from [sdpa_micro.cl:473-932](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L473-L932):

```
for each k_block:
    ┌─ KQ GEMM (DPAS)        — compute S = K^T × Q
    ├─ softmax                — max-reduce, exp, sum         (NO DPAS)
    ├─ S → SLM pack           — fp32→fp16 convert + store    (NO DPAS)
    ├─ barrier                — dead cycles
    ├─ rescale A_tile         — exp + multiply                (NO DPAS)
    ├─ VS GEMM (DPAS)        — compute A += V × softmax(S)
    └─ barrier                — dead cycles
```

Non-DPAS work includes:
- SLM atomic max across SG-rows ([sdpa_micro.cl:627-628](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L627-L628))
- Barrier + read-back of WG-wide max ([sdpa_micro.cl:693-694](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L693-L694))
- `native_vexp2` on every S element ([sdpa_micro.cl:700-704](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L700-L704))
- Reduction sums ([sdpa_micro.cl:719-721](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L719-L721))
- Half-precision pack + SLM store ([sdpa_micro.cl:734-740](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L734-L740))
- Previous A_tile rescale with exp ([sdpa_micro.cl:744-766](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L744-L766))

**2. Barriers with single-WG occupancy = dead DSS time**

At least 3 barriers per k-block iteration (lines [470](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L470), [629](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L629)/[693](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L693), [741](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L741)/[847](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L847)). With 1 WG/DSS, during a barrier the **entire DSS idles** — no second WG to fill in.

**3. SLM bandwidth contention**

Both GEMMs use SLM as B-operand source (Q from Q_slm, softmax(S) from S_slm). 32 SGs doing concurrent strided reads create bank conflicts → stall cycles that DPAS cannot fill.

**4. Occupancy amplifies factors 2 & 3**

4 threads/EU means only 3 alternates when one thread stalls on SLM bank conflict or barrier. With 8 threads/EU, 7 could cover.

**Rough cycle breakdown (single k-block, h256):**

| Phase | Est. cycles | DPAS? |
|---|---|---|
| KQ GEMM (128×128, k=256) | ~500 | Yes |
| Softmax + rescale + pack | ~290 | No |
| VS GEMM (256×128, k=128) | ~500 | Yes |
| Barriers + prefetch overhead | ~80 | No |
| **Total** | **~1370** | |
| **DPAS fraction** | **~73%** | |

After barrier stalls (amplified by 1 WG/DSS) and SLM contention → effective DPAS utilization lands around **50-55%** of roofline.

### Potential improvements

| Approach | Effect |
|---|---|
| Smaller WG (16 SGs) + 128-GRF | 2 WGs/DSS, 8 thr/EU → overlap barriers. Tradeoff: smaller tiles, less DPAS efficiency per call |
| Fused softmax (avoid S SLM round-trip) | Eliminates 1 barrier + SLM write/read. Hard with current 2-GEMM architecture |
| Flash-attention style tiling | Keeps S in registers, fuses softmax into GEMM loop at finer grain. Fundamentally different kernel design — attacks root cause (non-DPAS phases + barriers) |


### SG, work-item, SIMD, and register ownership

#### Hardware mapping

On Intel GPUs, 1 subgroup = 1 hardware thread = 1 SIMD execution unit. A subgroup of SUBGROUP_SIZE=16 work-items is **not** 16 independent entities — it is **one hardware thread** executing SIMD16. Every instruction operates on 16 lanes in lockstep:

```
                    ┌── Hardware Thread (= 1 Subgroup) ──┐
                    │                                     │
  SIMD16 instruction: ADD r4, r2, r3                     │
                    │                                     │
  Lane 0:  r4[0]  = r2[0]  + r3[0]   ← "work-item 0"   │
  Lane 1:  r4[1]  = r2[1]  + r3[1]   ← "work-item 1"   │
  ...                                                     │
  Lane 15: r4[15] = r2[15] + r3[15]  ← "work-item 15"  │
                    │                                     │
                    │  All 16 in ONE cycle, ONE instr.    │
                    └─────────────────────────────────────┘
```

#### Register ownership

**The thread (= subgroup) owns the GRF allocation**, not individual lanes. Each GRF entry = **64 bytes** = 16 dwords = one 32-bit value per SIMD lane:

```
GRF r4:  [ lane0:f32 | lane1:f32 | lane2:f32 | ... | lane15:f32 ]
          ←──────────────── 64 bytes ────────────────────────────→
```

"A_tile = 64 floats per lane" means 64 GRFs allocated to the thread, each holding 16 lanes:

```
  r100: [ A[0][q+0], A[0][q+1], ..., A[0][q+15] ]    ← head dim 0
  r101: [ A[1][q+0], A[1][q+1], ..., A[1][q+15] ]    ← head dim 1
  ...
  r163: [ A[63][q+0], A[63][q+1], ..., A[63][q+15] ]  ← head dim 63
```

#### GRF file partitioning

The GRF file is a **fixed pool per EU**, partitioned among resident threads:

| Mode | GRFs/thread | Threads/EU | Total GRFs/EU |
|---|---|---|---|
| 128-GRF | 128 | 8 | 1024 |
| 256-GRF | 256 | 4 | 1024 |

Same physical 1024 GRFs, just sliced differently. 256-GRF gives each thread more registers but halves thread count.

#### Mapping to `sdpa_micro__prefill` (h256, Xe2)

```
1 Workgroup = 32 Subgroups = 32 Hardware Threads
              distributed across 8 EUs → 4 threads per EU

Per EU:
  Thread 0 (SG  0): 256 GRFs  ← owns its A_tile, S_tile, etc.
  Thread 1 (SG  8): 256 GRFs  ← independent copy
  Thread 2 (SG 16): 256 GRFs  ← independent copy
  Thread 3 (SG 24): 256 GRFs  ← independent copy
  ─────────────────────────────
  Total: 1024 GRFs (full EU GRF file)
```

#### DPAS and threads

DPAS is a **thread-level** instruction. One thread issues one DPAS using its own registers:

```
dpas.8x8  rDst, rSrc0, rA, rB
```

This performs an 8×16 × 16 MAC. The other 3 threads on that EU can issue their own DPAS to pipeline the systolic array (filling stages while an earlier thread's result drains).


### Cycle breakdown methodology and limitations

The cycle estimates are **rough analytical models**, not measurements. Here is the derivation.

#### DPAS throughput model (Xe2, SIMD16, fp16→f32)

| Parameter | Value |
|---|---|
| Output per `dpas.8x8` instruction | Repeat(8) × SIMD(16) = **128 f32 results** |
| K consumed per instruction | Depth(8) × 2 (fp16 pair per 32-bit) = **16** |
| MACs per instruction | 8 × 16 × 16 = **2048** |
| Systolic latency | ~8 cycles |
| Issue rate (fully pipelined) | 1 DPAS/cycle per EU |

With 4 threads/EU in 256-GRF mode, pipeline fill is partial. Assumed **~1.5 cycles per DPAS** as middle estimate (range: 1.0 best-case to 2.0 worst-case).

#### KQ GEMM (h256): `S[128×128]`, inner dim k=256

Per subgroup (`sg_tile = 16×32`):
```
M steps: 16/8 = 2        (Repeat=8 covers 8 rows)
N steps: 32/16 = 2        (SIMD16 covers 16 cols)
K steps: 256/16 = 16      (each DPAS eats 16 along K)
Total: 2 × 2 × 16 = 64 DPAS instructions per SG
```

Per EU (4 threads): `4 × 64 = 256 DPAS instructions`. At ~1.5 c/instr → ~384 cycles.
Add ~20-30% for A/B operand loads (SLM reads for Q, global reads for K) partially overlapping: **~400-500 cycles**.

#### VS GEMM (h256): `A[256×128]`, inner dim k=128

Per subgroup (`sg_tile = 32×32`):
```
M steps: 32/8 = 4
N steps: 32/16 = 2
K steps: 128/16 = 8
Total: 4 × 2 × 8 = 64 DPAS instructions per SG    ← same count as KQ
```

Per EU: same math → **~400-500 cycles**.

#### Softmax + rescale phase

Counting operations in [sdpa_micro.cl:626-770](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L626-L770):

| Operation | Reference | Per-lane elems | Est. cycles |
|---|---|---|---|
| `tile_vreduce_max(S_tile)` | [L626](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L626) | 32 compares | ~10 |
| `tile_atomic_max_full` → SLM | [L627-628](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L627-L628) | 2 atomics | ~10 |
| **Barrier (arrive + wait)** | [L629,693](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L629) | | **~30-50** |
| `tile_load_full` (S_max from SLM) | [L694](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L694) | 2 floats | ~5 |
| `tile_vbroadcast_sub` (S -= max) | [L697](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L697) | 32 floats | ~8 |
| `native_vexp2` × scale | [L700-704](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L700-L704) | 32 floats | ~32 |
| `tile_vreduce_add` (col sum) | [L719-721](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L719-L721) | 32 floats | ~10 |
| `tile_copy_to_half2` (f32→f16) | [L734](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L734) | 32→16 | ~10 |
| `tile_store_t_sys_src2` → SLM | [L738-740](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L738-L740) | 16 uints | ~10 |
| **Barrier** | [L741,847](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L741) | | **~30-50** |
| Rescale: `exp_sub` on old max | [L751](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L751) | 2 floats | ~4 |
| `tile_binary(S_sum, old, mul)` | [L752](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L752) | 2 floats | ~2 |
| `tile_hbroadcast_mul(A_tile)` | [L766](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_micro.cl#L766) | 64 floats | ~16 |

**Softmax + rescale total: ~180-260 cycles.** Dominated by barriers (~60-100 cycles dead time with 1 WG/DSS) and element-wise exp2.

#### Combined estimate per k-block (h256)

| Phase | Low | High | DPAS? |
|---|---|---|---|
| KQ GEMM | 400 | 500 | Yes |
| Softmax + pack + rescale | 180 | 260 | No |
| VS GEMM | 400 | 500 | Yes |
| Prefetch overhead | 20 | 40 | No |
| **Total** | **1000** | **1300** | |
| **DPAS fraction** | **~70%** | **~75%** | |

After real-world barrier stalls (amplified by 1 WG/DSS) and SLM bank conflicts → **~50-55%** effective DPAS roofline.

#### Sources of estimation error

| Source | Direction | Impact |
|---|---|---|
| DPAS pipeline fill (4 threads, assumed 1.5c) | Could be 1.0 or 2.0 | ±30% on GEMM phases |
| Barrier cost (thread arrival skew) | Hard to predict | ±50% on barrier time |
| SLM bank conflicts | Not modeled | Understates non-DPAS |
| IGC instruction scheduling | Can hide ALU behind DPAS | Overstates non-DPAS |
| Prefetch hit rate | Determines GEMM stalls | Could increase GEMM time |

The DPAS fraction estimate should be read as **~65-80%**, not a precise 73%. For exact numbers, profile with `intel_gpu_top` or `ze_tracer` to measure actual DPAS active cycles vs total.
