# Continuous Batching: dynamic_split_fuse vs vLLM

## High-level contrast

- **dynamic_split_fuse=true**: prompt can be split across iterations; scheduler first prioritizes generation, then greedily schedules prompt chunks to fill the token budget. This allows long prompts to be partially scheduled under `max_num_batched_tokens` and interleaves multiple requests in later iterations.
- **dynamic_split_fuse=false (vLLM mode)**: prompt phase schedules whole prompt in a single shot (no splitting). If a prompt length exceeds the batch token limit, it cannot be scheduled. After prompt phase completes, generation phase schedules 1 token per active prompt per iteration.

## Scheduling behavior with multiple prompts

- **dynamic_split_fuse**: requests are pulled in order; prompt chunking is greedy by request order within the token budget. Early iterations can be dominated by Prompt0, with other prompts getting tokens only when capacity remains.
- **vLLM mode**: prompt phase schedules full prompts (up to max tokens per batch) in order; if capacity remains, next prompt is scheduled in the same iteration. Generation then proceeds with 1 token per prompt per step.

## dynamic_split_fuse=true (n=3) — per-iteration per-prompt input tokens

Large table moved to:

- [toolbox/.github/skills/dynamic_split_fuse_true_n3_tokens.tsv](toolbox/.github/skills/dynamic_split_fuse_true_n3_tokens.tsv)

## dynamic_split_fuse=false / vLLM mode (n=3) — per-iteration per-prompt input tokens

Large table moved to:

- [toolbox/.github/skills/vllm_n3_tokens.tsv](toolbox/.github/skills/vllm_n3_tokens.tsv)

# Contienous Batching logging practice

## Goals

- Capture per-iteration token scheduling from GenAI logs.
- Correlate each PagedAttention execution passage to kernel stages.
- Consolidate scheduler tokens, kernel stages, and output tokens in one audit table.

## Key commands and skills

- Enable verbose GPU/CB logging and run the benchmark to generate the trace.
- Extract execution passages using `prepare_primitive: Execute ... (type: paged_attention)` boundaries.
- Parse per-iteration prompt tokens from `CB_SCHEDULE iter=... req=... tokens=...` lines.
- Pull kernel names from `Execute stage = ...` lines within the same passage.
- Derive `pa_output_tokens` from the passage line `output tensor: f16:bfyx:<N>x4096:nopad`.
- Align scheduler records to the nearest unconsumed PagedAttention execution passage.

## Consolidated audit output

The consolidated correlation table is stored at:

- [toolbox/.github/skills/paged_attention_genai_kernel_correlation.tsv](toolbox/.github/skills/paged_attention_genai_kernel_correlation.tsv)

It contains one row per PagedAttention execution (`pa_exec_idx`), with correlated prompt tokens, kernel stages, and output token size.

Commandline:

```bash
cd /home/ceciliapeng/openvino.genai/build/bin && ./continuous_batching_benchmark --device GPU -m $HOME/llm_irs/WW05_llm-optimum_2026.0.0-20947/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT --cache_size 4 -n 2 --dataset /home/ceciliapeng/llm_irs/ShareGPT_V3_unfiltered_cleaned_split1.json
```

## enable_prefix_caching skills

### What it does

- Reuses previously computed KV blocks by prefix hash when `enable_prefix_caching=true`.
- Works for token and embedding inputs via `Sequence::get_hash()` prefix+block hashing.

### Important behavior

- Reuse is attempted at request admission via `restore_cached_blocks()`.
- Shared-prefix prompts can share KV blocks with ref-count / copy-on-write behavior.
- If cache is full, least-recently-used cached blocks are overwritten.

### Practical implications

- Best hit rate in multi-turn chat or repeated-prefix workloads.
- Simultaneous cold-start prompts with same prefix typically do not hit until one prompt has populated cache blocks.
- `clear_kv_cache()` is not allowed when prefix caching is enabled.

### References

- [toolbox/.github/skills/prefix_caching_hit_miss_timeline.md](toolbox/.github/skills/prefix_caching_hit_miss_timeline.md)
- [openvino.genai/src/cpp/src/continuous_batching/pipeline_impl.cpp](openvino.genai/src/cpp/src/continuous_batching/pipeline_impl.cpp#L302-L304)
- [openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp](openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp#L1105-L1158)
- [openvino.genai/src/cpp/src/sequence_group.cpp](openvino.genai/src/cpp/src/sequence_group.cpp#L12-L99)

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

GQA micro conditions (GENERATE stage):

- `kv_group_size == 8` and `k_head_size == 64`.
- When not met, micro SDPA is disabled for GENERATE.

Reference: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp#L1344-L1354)

### Kernel selection (micro SDPA disabled)

- PREFILL → `sdpa_opt__multi_tokens`
- MIXED → `paged_attention_opt__multi_tokens` (+ `paged_attention_opt__multi_tokens_finalization` when partitions > 1)
- GENERATE → `paged_attention_opt__single_token` or `paged_attention_opt__gqa_single_token` (+ finalization when partitions > 1)

Reference: [openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp](openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp#L1386-L1410)
