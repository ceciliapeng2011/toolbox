## Parsing rules (IMPORTANT)
- Ignore anything inside HTML comments: <!-- ... -->
- Ignore any section under a heading that starts with "DISABLED:"
- Never read or follow instructions labeled "DISABLED", "COMMENTED", or "BACKLOG", until it is explictly asked by another task.

## Model QWen3-8B
prompt len	1~32768
num_head	32
num_kv_head	8
head size	128

## BACKLOG: OpenVINO build
cd ~/openvino && rm ./build_Debug/src/plugins/intel_gpu/src/graph/impls/cm -rf && unset TBB_DIR && nohup ./build.ov.sh Debug ON OFF

## BACKLOG: OpenVINO.genai build
cd ~/openvino.genai && nohup ./build.sh

## BACKLOG: sync up aboutSHW and openvino CM PA kernels
source folder: aboutSHW/opencl/tests/pageatten/
destination folder openvino/src/plugins/intel_gpu/src/graph/impls/cm/
Compare files with the same file name (ignore suffix), copy code change from source folder to destination folder. 
1. Keep code lines of namespace and KERNEL_NAME unchanged.
2. cm_sdpa_2nd_reduce in aboutSHW/opencl/tests/pageatten/pa_single_token.cm is openvino/src/plugins/intel_gpu/src/graph/impls/cm/pa_single_token_finalization.cm

## BACKLOG: sync up aboutSHW and openvino CM Attention/X-Attn kernels
destination folder: aboutSHW/opencl/tests/x_attn/ AND aboutSHW/opencl/tests/pageatten/
source folder openvino/src/plugins/intel_gpu/src/graph/impls/cm/
Compare files with the same file name (ignore suffix), copy code change from source folder to destination folder.
1. Keep code lines of namespace and KERNEL_NAME unchanged.
2. Some JIT consts may change. Remove legacy ones in destination and use the updated ones from source.
3. Run test_kvcache_update.py test_pa_decoding.py and test_pa.py to validate pageatten test cases.
4. Run test_gemm_qk.py, test_xattn.py and test_post_proc.py to validate x_attn test cases.

## BACKLOG: report
1. Kernel prefixes like "pa_kv_cache_update", "pa_multi_", "pa_single_" are used by CM path.
2. Kernel prefixes like "pa_kv_cache_update", "sdpa_micro", "paged_attention_opt__" are used by OCL path.
3. Look these prefixes in each clintercept_trace.json, list the kernels and latency for each case.
4. To get kernel avg ms, you should do this way: for each type of kernel, get its avg ms; and sum up the avg ms of all types of kernel.
5. Write the comparision result into a csv file, arrange them by case id.

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



## COMMENTED: TASK 2: enable multi-sequence CM PA unit test
1. Read code aboutSHW/opencl/tests/pageatten/test_pa.py
2. Implement accuacy test for multi-sequence unit test

## COMMENTED: TASK 3: multi-seq support for CM PA
1. Learn workload split strategy of kernel "cm_sdpa_vlen" from /home/ceciliapeng/OCL/aboutSHW/opencl/tests/flashattn/cmfla.py.
2. Apply this strategy to kernel 

## COMMENTED: Task 4
1. Reference path must be Torch CPU only. Do not call `single_token_attention` or `multi_token_attention` kernels from reference.
2. Reference path must execute subsequences independently (split-by-subsequence). Runtime path in `PagedAttentionRunner::run` must stay packed/no-split and dispatch CM kernels only.
3. `PagedAttentionRunner::run` must never call `reference_attention` (or any reference helper). `single_token_attention` / `multi_token_attention` are CM GPU execution paths only.
4. All `generate_only` cases must route to `pa_single_token`.
5. OpenVINO path is single-subsequence only; do not introduce multi-subsequence assumptions there.
6. In kernel: a subsequence can span multiple WGs; but a WG cannot span multiple subsequences.

<!-- Review multi-subsequence kernels and workload scheduling under ~/OCL/aboutSHW/opencl/tests/pageatten, any accuracy problem? Any perfomrance optimization chance? -->

<!-- Should int causal_left = q_start + past_lens; or int causal_left = q_start; ?  inspect pa_multi_token.cm and the single-token path to confirm the same convention is used end-to-end.
Any causal masking issue left? -->

<!-- cd ~/OCL/aboutSHW/opencl/tests/pageatten && cliloader -cdt --dump-dir "./dump_generate_only" timeout 20s python -m pytest -q test_pa_multiseq.py -vv -k 'generate_only'
cliloader -cdt --dump-dir "./dump_mixed_only" timeout 20s python -m pytest -q test_pa_multiseq.py -vv -k 'mixed_only'
cliloader -cdt --dump-dir "./dump_prefill_only" timeout 20s python -m pytest -q test_pa_multiseq.py -vv -k 'prefill_only' -->

<!-- After rerun:

python -m py_compile ✅
pytest result: 8 failed, 11 passed
On your question: No, the failures are not only related to compressed_kvcache.

Evidence from failing IDs:

Non-compressed (cmpr0) failures:
generate: 1x34_1x515 (multiple configs)
mixed: 1x34_25_10x34 (multiple configs)
Compressed (cmpr1) failures:
generate: 1x34_1x515
mixed: 1x34_25_10x34
So compression contributes in some cases, but the mismatch exists in both compressed and non-compressed paths (mainly multi-sequence decode/mixed with multi-token path). -->

## COMMENTED: TASK 5 - adopt functionality from aboutSHW project to openvino project.
1.  For multi-subsequence batches, xattention estimation is conservatively disabled and the dense multi-token CM path is used. 
2.  No need to inspect unit test harness. Just kernel implementation change, kernel parameters set up, kernel workload and internal memory set up, etc. So ignore test cases and test_cb.py, perf test. 
3.  Now apply the multi-subsequence functionality of /home/ceciliapeng/OCL/aboutSHW/opencl/tests/pageatten to openvino/src/plugins/intel_gpu/src/graph/impls/cm/. 
4.  Add multi-seubsequence test cases into smoke_cm_xattention of src/plugins/intel_gpu/tests/unit/test_cases/paged_attention_gpu_test.cpp. Only extend cases for "std::vector<SubsequenceDescriptor> subsequences" with the same value from smoke_paged_attention, keep threshold 100 if it is multi-subsequences.

## TASK 6: Run xattention test cases till stable. 
1.  Write a test shell script run_cm_xattn_watchdog.sh under openvino path if it does not exist: The test is openvino/bin/intel64/Release/ov_gpu_unit_tests. Run xattention_test + smoke_cm_xattention_block_size and smoke_cm_xattention related cases only. 
Note: some cases may cause GPU hang, add a 1 minute watchdog to run each of them. If times out, kill the process, pick out and record the failure case, and continue to next case.
2.  Report all failures, fix each of them, and then re-run with the script iteratively, untill all cases pass.
3.  To save openvino building time, you can add the failure case to aboutSHW project opencl/tests/pageatten/test_pa_multiseq.py, and fix it there. If aboutSHW works but openvino not, then check the inputs, kernel parameters, workload sizing of both path. You can use OV_VERBOSE=4 to get openvino runtime log.
4.  Once it is fixed and verified with aboutSHW, apply the fix to openvino too.
5.  Make sure CM path is strictly used without fallback to OCL/micro kernels. 
6.  refer to "BACKLOG: OpenVINO build" to rebuild openvino if you change openvino code.
7.  Note "source ~/OCL/aboutSHW/ocl.venv/bin/activate" is for aboutSHW; "source /home/ceciliapeng/openvino.venv/bin/activate && OV0" is for openvino project.
8.  Once "TASK 6" is completed, start "TASK 7".
9.  To fix a failure case, do not read while folder of openvino, it is related to openvino/src/plugins/intel_gpu/src/graph/impls/cm/ only. Note: This strict “CM-only” restriction is applied when you are trying to read the whole source code of openvino. You can go beyond the restriction if it is not to read code, as usually the bug is only because of openvino/src/plugins/intel_gpu/src/graph/impls/cm/ code change. And you can go to aboutSHW for sure for cross check.

If you have a fix in cm kernels, do "cd ~/openvino && rm ./build_Debug/src/plugins/intel_gpu/src/graph/impls/cm -rf" before build openvino_intel_gpu_plugin , as cm kernels are not properly building-dependency checked.

## TASK 6.1:
Write another test shell script run_perf_watchdog.sh under openvino path if it does not exist: The test is openvino/bin/intel64/Release/ov_gpu_unit_tests. Use a argument to swith - 
1. One function for xattention_test + smoke_paged_attention_perf_cm related cases;
2. Another for paged_attention_test + smoke_paged_attention_perf_ocl related cases;
3. Both 1 and 2, and compare the performance into a csv file.
Note: some cases may cause GPU hang, add a 10 minutes watchdog to run each of them. If times out, kill the process, pick out and record the failure case, and continue to next case.
Reference to /home/ceciliapeng/openvino/run_2nd.sh to use "cliloader" to profile GPU kernels.

### COMMENTED: TASK 7: run CB multi-sequence task
#### prepare benchmark_cb_3cases.sh under openvino.genai path, if it does not exist.
cd ~/openvino.genai/ &&
./build/bin/continuous_batching_benchmark --device GPU -m $HOME/llm_irs/WW05_llm-optimum_2026.0.0-20947/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT --dataset /home/ceciliapeng/llm_irs/ShareGPT_V3_unfiltered_cleaned_split1.json --device_config ov.fp16.config --cache_size 4 --num_prompts 3 --max_input_len 4096 --max_output_len 2048 --max_batch_size 1024
Note: Use OV_GPU_PAGED_ATTENTION_IMPL=cm to force CM path which uses kvcache_block_size=16, use OV_GPU_PAGED_ATTENTION_IMPL=ocl to force OCL/micro path. We need compare both of them.
And also use "--use_xattention" along with "OV_GPU_PAGED_ATTENTION_IMPL=cm" to test kvcache_block_size=256. We need benchmark these 3 cases with a shell script under genai path.

Update benchmark_cb_3cases.sh: add an argument to switch running with cm, ocl, both (default); add another argument to switch multi, split mixed route mode, which works for cm cases only; add an argument to compare cases under temp folder and generate a report of csv, if any of the following ocl, cm+multi+blocksize 16, cm+split+blocksize 16, cm+multi+blocksize 256, ccm+multi+blocksize 256 log/profiling folders exist. All of these cases have a specific folder to capture their logs and profiling data under ./temp/ folder without date suffixed. "blocksize 256" is when use_xattention option is enabled.
#### run benchmark_cb_3cases.sh

## TASK 8: enable xattention
In aboutSHW/opencl/tests/x_attn/, enable multi-subsequences support.