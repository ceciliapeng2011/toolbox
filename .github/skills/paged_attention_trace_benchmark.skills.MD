---
name: Continuous Batching
description: PagedAttention CLIntercept Benchmark Skill.
---


## Goal
Benchmark `smoke_paged_attention_perf_ocl` and `smoke_paged_attention_perf_cm`, keep per-case CLIntercept traces, extract PA kernel timings, and compute speedups.

---

## Polished prompts

### Prompt A — extract PA kernels from all traces
Read all `clintercept_trace.json` under `dump/**`.
Build a table with columns:
1. case folder (`cm_basic_XX` / `ocl_basic_XX`)
2. kernel name
3. run index
4. duration (us)
5. duration (ms)

Only include kernels whose name starts with `sdpa_micro` or `pa_multi`.

### Prompt B — summarize per case
From the same traces, build a summary table grouped by `(case folder, kernel)` with:
- count
- min/avg/max in ms

### Prompt C — main PA speedup vs OCL baseline
For each `case_idx` (`0/1/2`), compare `cm_basic_XX` vs `ocl_basic_XX` using average latency.
Use OCL as baseline:
- speedup = `ocl_avg / cm_avg`
- latency reduction = `(1 - cm_avg / ocl_avg) * 100%`

### Prompt D — KV-cache kernel summary and speedup
Build the same summary and speedup tables for kernels starting with `pa_kv_cache`.
Use OCL as baseline per `case_idx`.

---

## `run.sh` as in-place command

```bash
pa_benchmark_trace.sh
```

---

## Speedup table 1: Main PA kernel (`sdpa_micro` / `pa_multi`)

CM uses block size 256 in this table.

| case_idx | ocl avg (ms) | cm avg (ms) | cm speedup vs ocl | cm latency reduction |
|---:|---:|---:|---:|---:|
| 0 | 38.953 | 21.039 | 1.85x | 45.98% |
| 1 | 93.141 | 43.630 | 2.13x | 53.15% |
| 2 | 134.289 | 57.372 | 2.34x | 57.28% |

## Speedup table 2: `pa_kv_cache*` kernel

| case_idx | ocl avg (ms) | cm avg (ms) | cm speedup vs ocl | cm latency reduction |
|---:|---:|---:|---:|---:|
| 0 | 0.657 | 0.608 | 1.08x | 7.46% |
| 1 | 0.642 | 0.536 | 1.20x | 16.51% |
| 2 | 0.686 | 0.582 | 1.18x | 15.16% |

## Speedup table 3: Forced impl comparison (`OV_GPU_PAGED_ATTENTION_IMPL`)

Caption: Different config from table 1 — OCL suite is forced with `OV_GPU_PAGED_ATTENTION_IMPL=OCL`; CM suite is forced with `OV_GPU_PAGED_ATTENTION_IMPL=CM` and `OV_GPU_USE_CM=ON`. In this setup, OCL runs `sdpa_micro__prefill` / `sdpa_micro__generate`, while CM runs `pa_multi_token_cm_bs1`. CM uses block size 16 in this table.

| case_idx | ocl avg (ms) | cm avg (ms) | cm speedup vs ocl | cm latency reduction |
|---:|---:|---:|---:|---:|
| 0 | 37.915 | 22.608 | 1.68x | 40.37% |
| 1 | 92.040 | 46.972 | 1.96x | 48.97% |
| 2 | 134.266 | 63.667 | 2.11x | 52.58% |

## Speedup table 4: OCL micro prefill `wg_tile_q=256`

Caption: Different config from table 3 — OCL micro prefill forced `wg_tile_q=256` (sdpa_micro prefill JIT/dispatch). OCL basic_1/2 failed with `CL_OUT_OF_RESOURCES` at this setting; only basic_0 is reported.

| case_idx | ocl avg (ms) | cm avg (ms) | cm speedup vs ocl | cm latency reduction |
|---:|---:|---:|---:|---:|
| 0 | 20.954 | 23.428 | 0.89x | -11.81% |
| 1 | N/A | N/A | N/A | N/A |
| 2 | N/A | N/A | N/A | N/A |

## Impact of CM block size change (256 -> 16)

Comparison of `cm avg (ms)` between table 1 (CM block size 256) and table 3 (CM block size 16):

| case_idx | cm avg (ms), bs=256 (table 1) | cm avg (ms), bs=16 (table 3) | delta (ms) | delta (%) |
|---:|---:|---:|---:|---:|
| 0 | 21.039 | 22.608 | +1.569 | +7.46% |
| 1 | 43.630 | 46.972 | +3.342 | +7.66% |
| 2 | 57.372 | 63.667 | +6.295 | +10.97% |

Conclusion: in these 3 cases, changing CM block size from 256 to 16 increases CM kernel latency by ~7.5% to ~11.0%.

## Speedup table 5: Forced impl comparison (`run_2nd.sh`, case 3~6)

Config used in `run_2nd.sh`:
- OCL runs: `OV_GPU_PAGED_ATTENTION_IMPL=ocl`, `cl_cache_dir=/home/ceciliapeng/.cache/cl_cache`, gtest `smoke_paged_attention_perf_ocl/paged_attention_test.basic/{3,4,5,6}`
- CM runs: `OV_GPU_PAGED_ATTENTION_IMPL=cm`, `cl_cache_dir=/home/ceciliapeng/.cache/cm_cache`, gtest `smoke_paged_attention_perf_cm/xattention_test.basic/{3,4,5,6}`
- Common settings: `cliloader -cdt`, separate `dump/ocl_basic_*` and `dump/cm_basic_*`, `--gtest_repeat=3 --gtest_color=no`

Kernel mapping used for table 5 main-PA latency extraction:
- OCL: `paged_attention_opt__*` (excluding `*finalization*`)
- CM: `pa_single_token_cm*`

| case_idx | ocl avg (ms) | cm avg (ms) | cm speedup vs ocl | cm latency reduction |
|---:|---:|---:|---:|---:|
| 3 | 0.556 | 0.254 | 2.19x | 54.31% |
| 4 | 2.097 | 0.958 | 2.19x | 54.31% |
| 5 | 4.148 | 1.753 | 2.37x | 57.74% |
| 6 | 8.381 | 3.335 | 2.51x | 60.20% |

Conclusion: in case 3~6 under forced implementation selection, CM remains consistently faster than OCL on the main PA kernel path, with speedup increasing from `2.19x` to `2.51x` as workload size grows.

---

## Notes
- Keep one dump subfolder per test case to avoid trace overwrite.
- Compare averages from equal repeat counts.
- Use same GPU frequency/governor conditions across runs for fair comparison.
