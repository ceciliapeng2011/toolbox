# MULTI_SUBSEQ_PREFILL_TO_PERF_MATRIX (2026-03-18)

## Scope
This skill captures the continuation history after the earlier summary milestone (`MULTI_SUBSEQ_PREFILL_FROM_9f42270f.skills.MD`) and documents how to drive the workflow from correctness checks to reproducible perf matrices for multi-subsequence paged attention.

---

## 1) User Intent Path (How to drive to this state)

### High-level route the user drove
1. Confirm causal semantics and path contracts.
2. Enforce OpenVINO single-subsequence constraints where required.
3. Patch causal masking in CM headers (XE2 then XE1), rebuild OpenVINO + openvino.genai.
4. Run/operate WWB + benchmark flow (including stop/restart scheduling behavior).
5. Shift fully to perf: design long-context QWen3-8B perf tests with no accuracy checks.
6. Add GPU-kernel timing using `cl.finish()` and separate perf types.
7. Expand perf matrix dimensions:
   - perf types: prefill / generate / mixed
   - block sizes: 16 / 256
   - cache compression: False / True
   - multiple subsequence templates per perf type
8. Produce full matrix and normalized takeaways.

### Core target requested by user
- Build a robust perf harness for paged attention in:
  - OCL/aboutSHW/opencl/tests/pageatten/test_pa_multiseq_perf.py
- Keep reference path constraints intact (no accidental accuracy/reference execution in perf path).
- Use GPU profiling timing from `cl.finish()` (not only host wall clock).
- Summarize a complete matrix and extract practical guidance.

---

## 2) Refined Prompts That Worked Best

Use these prompt patterns to reproduce this result quickly:

1. Causal semantics + convention
- "Check whether `causal_left` should be `q_start + past_lens` or `q_start`, and verify convention end-to-end."

2. Constraint lock-in
- "OpenVINO path only supports single-subsequence. Keep that contract."
- "Reference split only in reference path; run path must stay CM-only."

3. Perf harness creation
- "Design perf-only cases for QWen3-8B, long context up to 32K, trunk-gated at 4K, no accuracy checks."

4. Timing fidelity
- "Use `run_perf` and include `cl.finish` GPU kernel timing, not only host timing."

5. Coverage expansion
- "For each perf type, run block_size 16 and 256, and kv_cache_compression True/False."
- "Add multiple subsequence variants per perf type."

6. Reporting
- "Summarize all cases as a matrix and provide baseline-normalized speedups and takeaway."

---

## 3) Key Notes / Guardrails

### Polished execution constraints (Task 4)
1. Reference path must be Torch CPU only. Do not call `single_token_attention` or `multi_token_attention` kernels from reference.
2. Reference path must execute subsequences independently (split-by-subsequence). Runtime path in `PagedAttentionRunner::run` must stay packed/no-split and dispatch CM kernels only.
3. `PagedAttentionRunner::run` must never call `reference_attention` (or any reference helper). `single_token_attention` / `multi_token_attention` are CM GPU execution paths only.
4. All `generate_only` cases must route to `pa_single_token`.
5. OpenVINO path is single-subsequence only; do not introduce multi-subsequence assumptions there.

1. Single-token vs multi-token path
- `PagedAttentionRunner` route selection is based on packed token count vs subsequence count and token-count shape per subsequence.
- Single-token path launches two kernels per iteration (`cm_sdpa_2nd`, `cm_sdpa_2nd_reduce`).
- Multi-token path launches one kernel (`cm_page_attention`).

2. Timing interpretation
- `host_avg_ms`: includes host enqueue + synchronization window.
- `gpu_avg_ms`: derived from profiler samples returned by `cl.finish()`.
- Small host-vs-gpu gap is normal; large gap implies non-GPU overhead.

3. `kernels_per_iter`
- Must be set according to execution path:
  - single-token => 2
  - multi-token => 1
- Needed for correctly grouping raw per-kernel latency samples into per-iteration totals.

4. Label/data consistency
- Perf case names must match actual `subsequences` math (`num_tokens + past_len` context total).
- Example corrected: `single_ctx32k` must be `ss(4096, 28672)`.

5. Pytest mark hygiene
- Register custom mark `perf` to avoid `PytestUnknownMarkWarning`.

---

## 4) Implementation Deltas That Matter

### Files touched
- OCL/aboutSHW/opencl/tests/pageatten/test_pa_multiseq.py
- OCL/aboutSHW/opencl/tests/pageatten/test_pa_multiseq_perf.py
- OCL/aboutSHW/opencl/tests/pageatten/pytest.ini

### Structural improvements
1. Added perf execution APIs in runners:
- Multi-token `run_perf(...)` returns output + GPU profile data + elapsed ms + dispatch mode.
- Single-token `run_perf(...)` returns output + GPU profile data + elapsed ms.
- Top-level `PagedAttentionRunner.run_perf(...)` returns unified tuple including path and `kernels_per_iter`.

2. Refactored duplicated logic in `PagedAttentionRunner`
- Shared helper for kernel input preparation.
- Shared helper for path decision (`single_token` vs `multi_token`).

3. Expanded perf matrix generation
- Programmatic generation across:
  - perf types (`prefill_only`, `generate_only`, `mixed_only`)
  - subsequence templates (3 each)
  - block sizes (16, 256)
  - compression (0/1)
- Total cases: 36.

---

## 5) Metrics Definition Used

Per case output includes:
- `host_avg_ms`
- `gpu_avg_ms`
- `gpu_min_ms`
- `gpu_max_ms`
- `q_tok_s`
- `path`

Derived normalization used for comparison:
- Baseline per case-tag: `bs16/cmpr0`
- `gpu_speedup = baseline_gpu_ms / current_gpu_ms`
- `qps_speedup = current_q_tok_s / baseline_q_tok_s`

---

## 6) Final Matrix (from last full run)

Run status:
- 36 passed
- full matrix executed successfully

### A) Raw absolute matrix (first matrix)

#### prefill_only

| case | bs | cmpr | path | q_tokens | max_ctx | host_avg_ms | gpu_avg_ms | q_tok_s |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| single_ctx4k | 16 | 0 | multi_token/dynamic | 4096 | 4096 | 8.469 | 8.463 | 483646.2 |
| single_ctx4k | 16 | 1 | multi_token/dynamic | 4096 | 4096 | 10.172 | 10.165 | 402686.2 |
| single_ctx4k | 256 | 0 | multi_token/dynamic | 4096 | 4096 | 7.143 | 7.137 | 573425.7 |
| single_ctx4k | 256 | 1 | multi_token/dynamic | 4096 | 4096 | 8.575 | 8.567 | 477659.2 |
| single_ctx32k | 16 | 0 | multi_token/dynamic | 4096 | 32768 | 96.236 | 96.227 | 42561.8 |
| single_ctx32k | 16 | 1 | multi_token/dynamic | 4096 | 32768 | 130.214 | 130.204 | 31455.9 |
| single_ctx32k | 256 | 0 | multi_token/dynamic | 4096 | 32768 | 96.877 | 96.869 | 42280.3 |
| single_ctx32k | 256 | 1 | multi_token/dynamic | 4096 | 32768 | 118.163 | 118.154 | 34663.9 |
| two_seq_ctx32k | 16 | 0 | multi_token/dynamic | 4096 | 32768 | 98.745 | 98.736 | 41480.6 |
| two_seq_ctx32k | 16 | 1 | multi_token/dynamic | 4096 | 32768 | 133.999 | 133.990 | 30567.4 |
| two_seq_ctx32k | 256 | 0 | multi_token/dynamic | 4096 | 32768 | 100.315 | 100.306 | 40831.4 |
| two_seq_ctx32k | 256 | 1 | multi_token/dynamic | 4096 | 32768 | 121.726 | 121.716 | 33649.3 |

#### generate_only

| case | bs | cmpr | path | q_tokens | max_ctx | host_avg_ms | gpu_avg_ms | q_tok_s |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| single_ctx32k | 16 | 0 | single_token | 1 | 32768 | 2.576 | 2.570 | 388.1 |
| single_ctx32k | 16 | 1 | single_token | 1 | 32768 | 2.545 | 2.539 | 393.0 |
| single_ctx32k | 256 | 0 | single_token | 1 | 32768 | 1.488 | 1.481 | 672.1 |
| single_ctx32k | 256 | 1 | single_token | 1 | 32768 | 1.000 | 0.994 | 999.7 |
| two_seq_ctx16k | 16 | 0 | single_token | 2 | 16384 | 3.433 | 3.426 | 582.5 |
| two_seq_ctx16k | 16 | 1 | single_token | 2 | 16384 | 2.091 | 2.086 | 956.5 |
| two_seq_ctx16k | 256 | 0 | single_token | 2 | 16384 | 1.404 | 1.397 | 1424.6 |
| two_seq_ctx16k | 256 | 1 | single_token | 2 | 16384 | 0.833 | 0.826 | 2401.7 |
| three_seq_mixed_ctx | 16 | 0 | single_token | 3 | 32768 | 7.831 | 7.826 | 383.1 |
| three_seq_mixed_ctx | 16 | 1 | single_token | 3 | 32768 | 6.965 | 6.959 | 430.7 |
| three_seq_mixed_ctx | 256 | 0 | single_token | 3 | 32768 | 2.477 | 2.470 | 1211.2 |
| three_seq_mixed_ctx | 256 | 1 | single_token | 3 | 32768 | 1.567 | 1.561 | 1914.5 |

#### mixed_only

| case | bs | cmpr | path | q_tokens | max_ctx | host_avg_ms | gpu_avg_ms | q_tok_s |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| trunk4k_plus_gen32k | 16 | 0 | multi_token/dynamic | 4097 | 32768 | 86.894 | 86.890 | 47149.4 |
| trunk4k_plus_gen32k | 16 | 1 | multi_token/dynamic | 4097 | 32768 | 122.193 | 122.185 | 33528.8 |
| trunk4k_plus_gen32k | 256 | 0 | multi_token/dynamic | 4097 | 32768 | 87.907 | 87.904 | 46605.9 |
| trunk4k_plus_gen32k | 256 | 1 | multi_token/dynamic | 4097 | 32768 | 109.462 | 109.453 | 37428.6 |
| half_trunk_plus_gen | 16 | 0 | multi_token/dynamic | 2049 | 32768 | 53.339 | 53.335 | 38414.6 |
| half_trunk_plus_gen | 16 | 1 | multi_token/dynamic | 2049 | 32768 | 75.325 | 75.321 | 27202.1 |
| half_trunk_plus_gen | 256 | 0 | multi_token/dynamic | 2049 | 32768 | 53.664 | 53.660 | 38182.3 |
| half_trunk_plus_gen | 256 | 1 | multi_token/dynamic | 2049 | 32768 | 68.167 | 68.162 | 30058.4 |
| two_prefill_plus_gen | 16 | 0 | multi_token/dynamic | 2049 | 32768 | 54.269 | 54.265 | 37756.2 |
| two_prefill_plus_gen | 16 | 1 | multi_token/dynamic | 2049 | 32768 | 76.512 | 76.508 | 26780.0 |
| two_prefill_plus_gen | 256 | 0 | multi_token/dynamic | 2049 | 32768 | 55.020 | 55.016 | 37241.0 |
| two_prefill_plus_gen | 256 | 1 | multi_token/dynamic | 2049 | 32768 | 69.203 | 69.199 | 29608.5 |

### B) Normalized matrix (GPU speedup vs `bs16/cmpr0`)

#### prefill_only

| case | bs16/cmpr1 | bs256/cmpr0 | bs256/cmpr1 |
|---|---:|---:|---:|
| single_ctx4k | 0.833x | 1.186x | 0.988x |
| single_ctx32k | 0.739x | 0.993x | 0.814x |
| two_seq_ctx32k | 0.737x | 0.984x | 0.811x |

#### generate_only

| case | bs16/cmpr1 | bs256/cmpr0 | bs256/cmpr1 |
|---|---:|---:|---:|
| single_ctx32k | 1.012x | 1.735x | 2.586x |
| two_seq_ctx16k | 1.642x | 2.452x | 4.148x |
| three_seq_mixed_ctx | 1.125x | 3.168x | 5.013x |

#### mixed_only

| case | bs16/cmpr1 | bs256/cmpr0 | bs256/cmpr1 |
|---|---:|---:|---:|
| trunk4k_plus_gen32k | 0.711x | 0.988x | 0.794x |
| half_trunk_plus_gen | 0.708x | 0.994x | 0.782x |
| two_prefill_plus_gen | 0.709x | 0.986x | 0.784x |

---

## 7) Takeaway

1. Prefill-heavy and mixed workloads
- Compression (`cmpr1`) consistently regresses performance in this environment.
- `bs256/cmpr0` is near parity or slightly better than `bs16/cmpr0`.

2. Generate-only workloads
- Larger block size (`bs256`) yields major gains.
- `bs256/cmpr1` is strongest across all generate templates.

3. Practical default policy from this run
- Prefill/mixed default: favor `cmpr0`; `bs256` acceptable and often similar/better.
- Generate default: favor `bs256`, and test `cmpr1` as preferred fast path.

---

## 8) Reproduction Command

- `PA_PERF=1 timeout 1800s python -m pytest -s -q test_pa_multiseq_perf.py -m perf`

---

## 9) Checklist for Future Iterations

1. Keep case names consistent with actual sequence totals.
2. Keep `kernels_per_iter` aligned with path kernel count.
3. Compare both `gpu_avg_ms` and `q_tok_s`; use baseline-normalized matrix for decisions.
4. Re-run matrix whenever kernel dispatch policy or cache layout changes.
