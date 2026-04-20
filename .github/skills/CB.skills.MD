---
name: Continuous Batching
description: Knowledge about Continuous Batching used by OpenVINO GenAI.
---


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

## Benchmark metrics Q&A

- **Q: Refer to `openvino.genai/tools/continuous_batching/benchmark/continuous_batching_benchmark.cpp` and explain how `input_throughput`, `output_throughput`, `mean_ttft`, and `mean_tpot` are calculated.**  
	**A:**
	- `input_throughput` = total input tokens across all finished requests / benchmark duration in seconds.
	- `output_throughput` = total output tokens across all finished requests / benchmark duration in seconds.
	- Per sequence, `ttft` is measured from request start time to the first readable token.
	- Per sequence, `mean_tpot` is computed from inter-token read gaps after the first token: cumulative inter-token time divided by emitted-token count after the first token.
	- Per request (`GenerationInfo`), sequence-level `ttft`/`mean_tpot` are averaged over all sequence IDs in that request.
	- Final reported `mean_ttft` and `mean_tpot` are the average of per-request means over all requests tracked by `GenerationInfoCollector`.
	- Source references: metric accumulation/printing in `print_statistics()` and sequence timing updates in `SequenceInfo::update()`.

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


