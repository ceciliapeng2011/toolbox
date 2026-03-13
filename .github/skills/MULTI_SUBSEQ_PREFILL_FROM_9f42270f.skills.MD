## Skill: Enable multi-subsequence prefill from base commit 9f42270f83efccabeeec60733edcd93e28334fd7

### Objective
Make multi-subsequence prefill run correctly in one enqueue per round in the standalone PA harness, while keeping optimized sparse dispatch available.

### Starting point
- Repository path: /home/ceciliapeng/OCL/aboutSHW/opencl/tests/pageatten
- Base commit: `9f42270f83efccabeeec60733edcd93e28334fd7`
- Target test to reproduce/validate:
  - `python -m pytest -s test_pa_multiseq.py -k test_pa_multiseq_pytest_prefill_batch_size_in_sequences_2[2-1-32-32-False-64]`

### Prompt to drive the implementation
Use this prompt in a coding agent session:

Implement true multi-subsequence prefill support in the page attention harness starting from commit 9f42270f83efccabeeec60733edcd93e28334fd7.

Requirements:
1) Runtime/scheduler behavior
- Keep scheduler rounds as multi-subsequence (do not serialize each subsequence into separate rounds).
- Submit one kernel enqueue per round with all subsequences packed.
- Keep API cleanup: remove legacy `seq_lens`/`trunk_sz` dependency; use `max_num_batched_tokens`.

2) Kernel mapping behavior
- In `pa_multi_token.cm`, map each workgroup to exactly one subsequence segment.
- Use `subsequence_begins` to find subsequence ownership of `wg_id`.
- Use `block_indices_begins[subsequence_id]` as base for KV block lookup.
- Use subsequence-local query offset for causal math (`q_start_local`) and keep global offset only for Q/O memory addressing.

3) Workgroup sizing and counts
- Compute `max_subsequence_q_len` first.
- Compute `wg_size = min(ceil_div(max_subsequence_q_len, q_step), max_wg_size)`.
- Do not cap `wg_count`; compute total `wg_count = sum(ceil_div(subseq_q_len, wg_seq_len))`.
- Use `gws = [1, num_heads, wg_count * wg_size]`, `lws = [1, 1, wg_size]`.

4) Hybrid dispatch
- Build two kernel variants in runtime:
  - dynamic variant (runtime `wg_seq_len`)
  - optimized variant (compile-time `CMPA_WG_SEQ_LEN`) only for sparse block sizes 128/256
- Select optimized variant only when conditions match (sparse in 128/256, `wg_size == max_wg_size`, and optimized seq-len matches sparse block size), otherwise use dynamic variant.

5) Macro/build-option cleanup
- Remove `DCMPA_USE_RUNTIME_WG_SEQ_LEN`.
- Pass `CMPA_WG_SEQ_LEN` only for optimized build options.
- In kernel, choose:
  - compile-time `CMPA_WG_SEQ_LEN` if defined and > 0
  - else `local_size * q_step`.
- In helper header (`cm_pa_xe2.hpp`), do not hard error when `CMPA_WG_SEQ_LEN` is absent; allow default-disabled optimized path.

6) Tests and diagnostics
- Convert tests to pytest-style parameterized cases.
- Add/keep multisequence prefill tests for two-session cases.
- Keep detailed enqueue logs during bring-up (dispatch mode, gws/lws, shapes, vectors), then optionally gate behind a debug switch.
- Validate at minimum with:
  - `test_pa_multiseq_pytest_prefill_batch_size_in_sequences_2[2-1-32-32-False-64]`

Acceptance criteria:
- Selected test passes with one enqueue for multi-subsequence rounds.
- Numerical check vs reference passes (`check_close`).
- Dynamic path builds without requiring compile-time `CMPA_WG_SEQ_LEN`.
- Optimized path remains buildable/usable when enabled.

### Code design ideas (why this worked)
1. Preserve scheduler semantics, fix kernel ownership
- The key is to keep scheduler producing multi-subsequence rounds and fix kernel-side WG-to-subsequence mapping, instead of flattening behavior back to single-subsequence compatibility.

2. Separate address spaces for correctness
- Global query offsets are for memory layout.
- Subsequence-local offsets are for causal-window math and per-subsequence bounds.
- Mixing them causes subtle correctness drift.

3. Use max-length-driven local size
- Decide `wg_size` from max subsequence length in the round to avoid under-utilization.
- Sum WG demand across subsequences for global dispatch coverage.

4. Keep fast path optional
- Hybrid build avoids forcing compile-time assumptions onto dynamic workloads.
- Optimized sparse mode is selected only when runtime shape/layout satisfies strict expectations.

5. Prefer permissive macro contracts
- Optional `CMPA_WG_SEQ_LEN` avoids unnecessary compile failures in dynamic mode.
- Build-time options should describe capability, not block unrelated execution modes.

### Files typically touched
- /home/ceciliapeng/OCL/aboutSHW/opencl/tests/pageatten/test_pa_multiseq.py
- /home/ceciliapeng/OCL/aboutSHW/opencl/tests/pageatten/pa_multi_token.cm
- /home/ceciliapeng/OCL/aboutSHW/opencl/tests/pageatten/cm_pa_xe2.hpp

### Practical checklist
- Reproduce failure on base commit.
- Implement runtime one-enqueue + WG sizing/count updates.
- Implement kernel subsequence mapping + local causal offset fix.
- Add hybrid dynamic/optimized kernel creation and selection.
- Remove `DCMPA_USE_RUNTIME_WG_SEQ_LEN`, keep `CMPA_WG_SEQ_LEN` optimized-only.
- Run target pytest selector and confirm pass.
- Run broader pytest matrix for regression confidence.
