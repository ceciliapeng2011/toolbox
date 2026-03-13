## Parsing rules (IMPORTANT)
- Ignore anything inside HTML comments: <!-- ... -->
- Ignore any section under a heading that starts with "DISABLED:"
- Never read or follow instructions labeled "DISABLED", "COMMENTED", or "BACKLOG", until it is explictly asked by another task.

## Model QWen3-8B
prompt len	1~32768
num_head	32
num_kv_head	8
head size	128

## FlashAttention and FlashDecoding
1. There are two matmul operations with int8 kvcache: gemm1 and gemm2.
2. A typical compute capability utilization of FlashAttention is 55%.
3. q_len is a part of input tokens of a sequence. It is decided by trunk size and Contineous batching scheduling policy, and kvcache budget.
4. q_len is scheduled among workgroups. Each workgroup may handle, e.g. 256 q tokens. So kvcache may be load multiple times - num_repeats_load_kvcache = q_len / 256.

## Contineous batching
A typical trunksize	is 4096.

## Problem statement of a PagedAttention workload
Suppose there are -
1. 'S' : batch_size_in_sequences - total number of sequences to execute. 's' in range [0, S-1]
2. 'q_len[s]' : Each sequence 's' has prompt_len – per sequence size number of scheduled tokens. q_len is in rage [1, trunksize]
3. 'kv_len[s]' : context_lens, kv_cache sequence lengths of each sequence. kv_len is in range [0, 32k]
4. As the initiate stage, ignore xattention and xattenion_block_size, only keep -DSPARSE_BLOCK_SIZE=1"

## COMMENTED: backup
1. enable multi sequence support for CM path: PD mixed, multiP, multiD
2. improve sdpa_micro_prefill - x2 with `wg_tile_q=256`
3. improve sdpa_micro_decode - `wg_tile_q=256` not currently supported by the selected microkernel/kernel-template constraints.
4. improve spda_opt_decode - x2
5. Use multiple immediate command lists (or SYCL queues) to allow `concurrent execution of context and generation kernels` if the driver/hardware can overlap them.
6. building a PagedAttention primitive that must drive mixed prefill (compute‑bound) and decode (memory‑bound) subsequences on an Intel iGPU (via SYCL/Level Zero). A practical design is Two-kernel strategy: one tuned for prefill, one for decode and Host‑side scheduler for mixed subsequences.

## Task 0
1. Read 'Paged Attention specification' from PA_spec skills.
2. Read 'Continuous Batching' skills from CB.skills.
<!-- 3. Read iGPU skills from GPU.skills. -->
<!-- 4. Read source code of openvino/src/plugins/intel_gpu/, especially the code about paged_attention. -->
<!-- 5. Read source code of openvino.genai, especially the code about contineous batching. -->

## COMMENTED: Task 1 : solution investigation for mixed P/D
1. How would you suggest to implment CM kernel to schedule a PagedAttention workload?

### Draft answer: CM kernel scheduling for PagedAttention

#### 1) Scheduling target
- Primary objective: minimize total execution time of the PagedAttention operator for each CB scheduling step.
- Secondary objective: maximize EU occupancy while minimizing repeated KV-cache loads.
- Prefer sequence-major work assignment with dynamic token blocks to handle variable `q_len[s]` and `kv_len[s]`.
- Treat `q_len[s]` as a roofline driver:
	- small `q_len[s]` -> memory-bound tendency,
	- large `q_len[s]` -> compute-bound tendency.

#### 1.1) Optimization metric (what to minimize)
- End-to-end operator time:
	$$T_{op}=T_{queue\_build}+T_{dispatch}+T_{kernel\_A}+T_{kernel\_B}+T_{reduce}+T_{writeback}$$
- Tuning decisions (`Q_TILE`, `KV_TILE`, split-KV, subgroup shape) should minimize `T_op`, not any single sub-metric in isolation.

#### 2) Core idea: two-level tiling + persistent workgroups
- Use 2D work decomposition:
	- `Q_TILE` over query tokens (recommended start: 128 or 256).
	- `KV_TILE` over paged KV blocks (recommended start: page-aligned, e.g. 64/128 tokens equivalent).
- Launch persistent threadgroups (workgroups) that pull tasks from a global queue using atomic fetch-add.
- Each task unit:
	- `(sequence_id s, q_begin, q_count, kv_page_begin, kv_page_count, head_group)`
- Benefit:
	- Better load balance for highly skewed `q_len[s]`/`kv_len[s]`.
	- Less tail idle than static 1-sequence-per-WG mapping.

#### 3) Task queue format (host-prepared)
- Build compact descriptors on host before dispatch:
	- `seq_id`
	- `q_start`, `q_len_tile`
	- `kv_start_page`, `kv_num_pages`
	- `head_start`, `head_count`
	- pointers/offsets to block tables (paged KV indirection)
- Sort descriptors by `(kv_num_pages desc, q_len_tile desc)` to reduce long-tail bubbles.

#### 4) Kernel pipeline (FlashDecoding style)
- **Stage A: QK^T partial + max/sum stats**
	- Load Q tile to SLM/registers once.
	- Stream K pages in `KV_TILE` chunks.
	- INT8 KV dequant in-kernel with per-block/per-channel scales.
	- Compute running softmax stats using online stable formula:
		- keep per-row `(m_i, l_i)` where
		$$m_i^{new}=\max(m_i, m_i^{tile}), \quad l_i^{new}=l_i\cdot e^{m_i-m_i^{new}}+l_i^{tile}\cdot e^{m_i^{tile}-m_i^{new}}$$
- **Stage B: PV accumulation**
	- Re-stream V pages in same traversal order.
	- Apply normalized weights from online stats and accumulate output.
- Optional fused single-pass variant when register pressure allows; otherwise keep two-pass for occupancy.

#### 5) Mapping policy
- Primary mapping unit: `(head_group, q_tile)`.
- Secondary split by KV pages when `kv_len[s]` is large:
	- enable split-KV reduction mode beyond threshold (e.g. `kv_len > 8k`).
	- reduce partial outputs via lightweight epilogue kernel.
- For short contexts (`kv_len <= 1k`), disable split-KV to avoid reduction overhead.
- Choose split-KV only when expected parallel gain exceeds reduction cost in `T_op`.

#### 6) Heuristics (initial)
- `Q_TILE`:
	- 256 for prefill-heavy (`q_len` large),
	- 64/128 for decode-heavy (`q_len` small) to improve latency.
- Bound-aware mode switch by `q_len` bucket:
	- **memory-bound region (small `q_len`)**:
		- prioritize memory efficiency: larger `KV_TILE` (if SLM allows), deeper software prefetch, fewer split/reduction passes.
		- reduce per-task overhead: coarser descriptor packing and fewer atomics.
	- **compute-bound region (large `q_len`)**:
		- prioritize math throughput: larger `Q_TILE`, higher subgroup concurrency, and fused epilogue when register budget permits.
		- allow split-KV more aggressively to increase parallel math occupancy.
- `KV_TILE`:
	- fit dequantized K/V fragments + score fragments in SLM budget.
- Group size:
	- start 8 or 16 SIMD lanes per subgroup; 2–4 subgroups per WG based on GRF pressure.
- Queue depth:
	- target at least `4x` number of resident WGs to hide imbalance.
- Autotune should learn the crossover `q_len*` per platform and switch policy at runtime.

#### 7) Continuous batching integration
- Rebuild descriptor queue each scheduling step (per trunk window).
- Keep stable sequence ordering only at final output writeback; internal compute order can be arbitrary.
- Support sequence churn (new, finished, evicted) by regenerating only changed descriptors.

#### 8) Memory/layout recommendations
- Store paged KV in page-contiguous layout to minimize gathers:
	- `[page][kv_head][token_in_page][head_dim_chunk]`
- Align page metadata and scales to cacheline boundaries.
- Precompute page-table offsets on host to avoid pointer chasing in hot loops.

#### 9) Correctness/perf guardrails
- Always apply causal + block mask before softmax stats update.
- Validate numeric drift of online softmax vs reference FP32 path.
- Add fallback kernel for extreme shapes (very small `q_len`, very large `kv_len`) when occupancy collapses.

#### 10) Minimal implementation plan
1. Implement host descriptor builder + atomic task queue.
2. Implement Stage A/B CM kernels with online softmax stats.
3. Add split-KV optional path + reduction kernel.
4. Add autotune table keyed by `(q_len bucket, kv_len bucket, num_heads, head_dim)`.
5. Benchmark on representative CB mixes (short decode + long prefill mixed batch).
6. Use measured `T_op` as objective for policy selection and runtime switch thresholds.

#### 11) Expected gain
- Higher utilization than static scheduling by reducing load imbalance.
- Lower redundant KV traffic via tiled reuse per `q_tile`.
- Better robustness across mixed `q_len[s]`/`kv_len[s]` continuous-batching workloads.



## TASK 2: enable multi-sequence CM PA unit test
1. Read code aboutSHW/opencl/tests/pageatten/test_pa.py
2. Implement accuacy test for multi-sequence unit test

## COMMENTED: TASK 3: multi-seq support for CM PA
1. Learn workload split strategy of kernel "cm_sdpa_vlen" from /home/ceciliapeng/OCL/aboutSHW/opencl/tests/flashattn/cmfla.py.
2. Apply this strategy to kernel 