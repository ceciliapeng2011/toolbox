---
name: SDPA Prefill Ratio + Long Prompt Predict
description: Summarize measured SDPA/prefill ratios and prediction method for 64k/128k/256k.
---

## Goal
Create a compact record of:
- measured ratios from [sdpa_prefill_ratio_using_user_avg_prefill.csv](../../CLIntercept_Dump/qwen3_next_388H_profile/sdpa_prefill_ratio_using_user_avg_prefill.csv)
- predicted long-prompt values from [predicted_long_prompts_attention_x2_others_x.csv](../../CLIntercept_Dump/qwen3_next_388H_profile/predicted_long_prompts_attention_x2_others_x.csv)
- exact method used to generate both.

---

## Measured data summary (`sdpa_prefill_ratio_using_user_avg_prefill.csv`)

Rule used:
- `avg_attention_per_prefill_ms = sum(sdpa_micro__prefill.dur) / 4 / 1000`
- `ratio(%) = avg_attention_per_prefill_ms / avg_prefill_ms_user * 100`

| seq_len | avg_prefill_ms_user | total_sdpa_prefill_ms | avg_attention_per_prefill_ms | attention_vs_prefill_pct | note |
|---:|---:|---:|---:|---:|---|
| 1k | 1009.350 | 42.775 | 10.694 | 1.059% | |
| 2k | 1857.310 | N/A | N/A | N/A | file_not_found |
| 4k | 3447.900 | 683.425 | 170.856 | 4.955% | |
| 8k | 6966.310 | 2715.461 | 678.865 | 9.745% | |
| 16k | 38964.840 | 12953.985 | 3238.496 | 8.311% | |
| 32k | 52092.770 | 56207.651 | 14051.913 | 26.975% | |

---

## Predicted data summary (`predicted_long_prompts_attention_x2_others_x.csv`)

Last-prompt model:
- `attention_ms` is $O(x^2)$
- `others_ms` is $O(x)$
- `prefill_ms = attention_ms + others_ms`

Fitted forms:
- `attention_ms = a * x^2`, where `a = 13.647638181`
- `others_ms = b * x`, where `b = 1361.745275533`
- `x` is prompt length in k tokens

| seq_len_k | pred_attention_ms | pred_others_ms | pred_prefill_ms | pred_attention_vs_prefill_pct |
|---:|---:|---:|---:|---:|
| 64 | 55900.726 | 87151.698 | 143052.424 | 39.077% |
| 128 | 223602.904 | 174303.395 | 397906.299 | 56.195% |
| 256 | 894411.616 | 348606.791 | 1243018.406 | 71.955% |

---

## How `sdpa_prefill_ratio_using_user_avg_prefill.csv` was generated

Input:
- folder: `/home/ceciliapeng/CLIntercept_Dump/qwen3_next_388H_profile/`
- user-provided prefill averages (ms):
  - 1k: 1009.35
  - 2k: 1857.31
  - 4k: 3447.9
  - 8k: 6966.31
  - 16k: 38964.84
  - 32k: 52092.77

Processing for each file `clintercept_trace_<k>.json`:
1. Load JSON list.
2. Keep events with `ph == "X"` and `name` containing `sdpa_micro__prefill`.
3. Sum `dur` as `total_sdpa_prefill_us`.
4. Compute `avg_attention_per_prefill_us_div4 = total_sdpa_prefill_us / 4`.
5. Convert to ms and compute ratio against provided `avg_prefill_ms_user`.
6. Write one CSV row; if file missing, mark `note=file_not_found`.

---

## How prediction was generated (including last prompt)

Prompt intent:
- “`attention_ms` is $O(x^2)$, other kernels are $O(x)$, and `prefill_ms = attention_ms + others_ms`.”

Procedure:
1. Use measured `avg_attention_per_prefill_ms` points for 1k/4k/8k/16k/32k.
2. Fit `attention_ms = a * x^2` through origin via least squares.
3. Compute measured `others_ms = prefill_ms_user - attention_ms_measured` on shared points.
4. Fit `others_ms = b * x` through origin via least squares.
5. Predict at 64k/128k/256k:
   - `pred_attention_ms = a * x^2`
   - `pred_others_ms = b * x`
   - `pred_prefill_ms = pred_attention_ms + pred_others_ms`
   - `pred_ratio = pred_attention_ms / pred_prefill_ms * 100`

---

## Reusable prompt snippets

### Build measured ratio CSV
“Summarize all `sdpa_micro__prefill` in each trace file. Divide by 4 to get avg attention per prefill. Use my provided avg prefill values and output ratio CSV.”

### Build prediction CSV (last model)
“Predict 64k/128k/256k with: `attention_ms` as $O(x^2)$, `others_ms` as $O(x)$, and `prefill_ms=attention_ms+others_ms`. Output CSV with attention, others, prefill, ratio.”

---

## What-if: CM / xattention on attention kernel

Assumption from prompt:
- CM path cuts attention duration by half.
- xattention algorithm cuts attention duration to a quarter.

Let:
- `P = avg_prefill_ms_user`
- `A = avg_attention_per_prefill_ms`
- `others = P - A`

Then:
- `prefill_cm = others + 0.5*A = P - 0.5*A`
- `prefill_xattention = others + 0.25*A = P - 0.75*A`

Combined measured + predicted table:

| source | seq_len | P (ms) | A (ms) | prefill_cm (ms) | prefill_xattention (ms) | cm_vs_prefill_pct | xattention_vs_prefill_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| measured | 1k | 1009.350 | 10.694 | 1004.003 | 1001.329 | 99.470% | 99.205% |
| measured | 4k | 3447.900 | 170.856 | 3362.472 | 3319.758 | 97.522% | 96.283% |
| measured | 8k | 6966.310 | 678.865 | 6626.877 | 6457.161 | 95.128% | 92.691% |
| measured | 16k | 38964.840 | 3238.496 | 37345.592 | 36536.968 | 95.844% | 93.769% |
| measured | 32k | 52092.770 | 14051.913 | 45066.814 | 41553.835 | 86.513% | 79.769% |
| predicted | 64k | 143052.424 | 55900.726 | 115102.061 | 101126.879 | 80.461% | 70.692% |
| predicted | 128k | 397906.299 | 223602.904 | 286104.847 | 230204.121 | 71.903% | 57.854% |
| predicted | 256k | 1243018.406 | 894411.616 | 795812.598 | 572209.694 | 64.023% | 46.034% |
| predicted | 512k | 4274860.044 | 3577646.463 | 2486036.813 | 1591625.197 | 58.155% | 37.232% |
| predicted | 1024k | 15705013.015 | 14310585.853 | 8549720.089 | 4972073.625 | 54.439% | 31.659% |

`2k` has no trace file (`file_not_found`), so no measured `A` was available in the CSV-based path.


## Take-aways for QWen3-next on PTL-388H
1. Starting at 32k, attention already accounts for a substantial share of TTFT (~27%). For longer prompts, the share is projected to rise further: ~39% at 64k, ~56% at 128k, and ~71% at 256k.
2. Switching to the CM + xattention path is expected to improve TTFT by roughly 14%–20% at 32k, and by about 20%–50% for longer prompts.
