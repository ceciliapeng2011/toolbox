# enable_prefix_caching: hit/miss timeline (one-page)

## Scope

This timeline describes one request entering Continuous Batching with `enable_prefix_caching=true`, and how prefix-cache hit/miss is resolved block-by-block.

## Timeline

### T0 — Pipeline setup / defaults

- `SchedulerConfig::enable_prefix_caching` exists and is logged in config.
- LLMPipeline latency-oriented defaults set it to `true` unless overridden.

References:
- [openvino.genai/src/cpp/include/openvino/genai/scheduler_config.hpp](openvino.genai/src/cpp/include/openvino/genai/scheduler_config.hpp#L62)
- [openvino.genai/src/cpp/src/utils.cpp](openvino.genai/src/cpp/src/utils.cpp#L735-L739)
- [openvino.genai/src/cpp/src/llm/pipeline.cpp](openvino.genai/src/cpp/src/llm/pipeline.cpp#L212-L220)

### T1 — Request arrives

- `add_request()` builds `SequenceGroup`.
- If prefix caching is enabled, pipeline immediately calls `restore_cached_blocks()` before normal scheduling.

Reference:
- [openvino.genai/src/cpp/src/continuous_batching/pipeline_impl.cpp](openvino.genai/src/cpp/src/continuous_batching/pipeline_impl.cpp#L302-L304)

### T2 — Prefix probe loop in `restore_cached_blocks()`

For each logical block span up to prompt length:

1. Compute hash for full block boundary (`content_len += block_size`).
2. Try `get_cached_block(full_block_hash, ...)`.

Reference:
- [openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp](openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp#L1120-L1123)

### T3 — HIT path (full-block hit)

- Cached per-layer blocks are returned.
- Blocks are stamped with new timestamp and appended to sequence block table.
- `processed_tokens` is advanced to reused prefix (or `prompt_len - 1` at full prompt boundary).
- Loop continues to attempt the next block.

References:
- [openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp](openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp#L1124-L1132)
- [openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp](openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp#L1131-L1132)

### T4 — MISS path (full-block miss)

- Fallback checks partial hashes within the current block (`prev + i`, `i in [1..block_size-1]`).
- If partial hit exists, restore one partially filled block and advance processed tokens accordingly.
- If no partial hit, stop restoring; remaining prompt is computed normally during prefill.

References:
- [openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp](openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp#L1134-L1149)
- [openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp](openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp#L1151-L1153)

### T5 — Scheduler runs with reduced prefill

- Scheduler sees some prompt tokens already processed.
- Only remaining prompt suffix is scheduled for prefill, then generation proceeds.

Reference:
- [openvino.genai/src/cpp/src/continuous_batching/scheduler.hpp](openvino.genai/src/cpp/src/continuous_batching/scheduler.hpp#L98-L116)

### T6 — During/after generation: cache population and reuse pool behavior

- New blocks are allocated with hash keys derived from prefix+block content.
- When blocks are freed:
  - with prefix caching ON: blocks go to hash store (`OverwritableBlocksHashStore`) for future restore;
  - if free pool exhausted later, least-recently-used cached block may be overwritten.

References:
- [openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp](openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp#L749-L754)
- [openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp](openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp#L332-L369)
- [openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp](openvino.genai/src/cpp/src/continuous_batching/block_manager.hpp#L441-L457)

### T7 — Hash semantics (why a hit is valid)

- `Sequence::get_hash()` recursively chains prefix hash and current block content.
- Same prefix + same block content => same hash => eligible for KV-block reuse.
- Works for token inputs and embeddings (embeddings are reduced before hashing).

References:
- [openvino.genai/src/cpp/src/sequence_group.cpp](openvino.genai/src/cpp/src/sequence_group.cpp#L12-L71)
- [openvino.genai/src/cpp/src/sequence_group.cpp](openvino.genai/src/cpp/src/sequence_group.cpp#L83-L99)

## Quick readout: hit vs miss outcomes

- Hit-heavy case: later turns with large shared history, low TTFT, small prefill token count.
- Miss-heavy case: new or diverged prefixes, normal prefill cost.
- Mixed case: full-block misses but partial-block hit recovers part of prompt.

## Operational notes

- Prefix caching forbids `clear_kv_cache()` assertions in scheduler path.
- Dynamic-allocation cleanup in pipeline clears KV cache only when prefix caching is OFF.

References:
- [openvino.genai/src/cpp/src/continuous_batching/scheduler.hpp](openvino.genai/src/cpp/src/continuous_batching/scheduler.hpp#L172-L176)
- [openvino.genai/src/cpp/src/continuous_batching/pipeline_impl.cpp](openvino.genai/src/cpp/src/continuous_batching/pipeline_impl.cpp#L628-L631)
