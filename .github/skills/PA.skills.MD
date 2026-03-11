## PagedAttention stage and kernel selection skills

### Stage decision (Intel GPU)

- `PagedAttentionStage::GENERATE` when `query_shape[0] == past_lens_shape[0]`.  【i.e. `batch_size_in_tokens == batch_size_in_sequences`, which means every sequence has 1 input token in query.】
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
