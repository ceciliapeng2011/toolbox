---
name: PA Spec
description: API design for PagedAttention.
---


## Paged Attention specification summary (for CB)

### Basic terms

- `batch_size_in_sequences`: number of active sequences in current PA call.
- `prompt_lens`: scheduled token count per sequence (current step/chunk).
- `batch_size_in_tokens = sum(prompt_lens)`.
- `max_context_len = max(context_lens)`.
- `max_num_blocks = div_up(max_context_len, BLOCK_SIZE)`.
- `num_blocks`: total entries in `block_indices` for the current batch.

### Core requirements

- Compute multiple prompts **without padding**.
- Use flattened token batch (`[batch_size_in_tokens, ...]`) instead of `[N, S, ...]`.
- Support mixed execution: prompt chunks and decode tokens in the same call.
- Support per-sequence variable `prompt_len`.
- Support speculative decode, chunked prefill, and partial recompute after eviction.
- Support cases where both `prompt_len > 1` and `context_len > 1` (not only pure prefill/decode).
- Support KV eviction where block tables can differ per decoder layer (layer-aware slot/block mapping).

### inputs semantic summary
#### Main PA inputs

- `[0] query`: `[batch_size_in_tokens, num_heads * head_size]`
- `[1] key`: `[batch_size_in_tokens, num_kv_heads * head_size]`
- `[2] value`: `[batch_size_in_tokens, num_kv_heads * head_size]`
- `[3] key_cache`: `[num_blocks, num_kv_heads, block_size, head_size]`
- `[4] value_cache`: `[num_blocks, num_kv_heads, block_size, head_size]`
	- Note: cache inner layout can be HW/plugin specific.

#### Sequence state inputs

- `[5] past_lens` (`i32`, `[batch_size_in_sequences]`):
	- per-sequence past KV length.
	- if `prompt_len > 1`, corresponds to the first token in that sequence subgroup.

- `[6] subsequence_begins` (`i32`, `[batch_size_in_sequences + 1]`):
	- start offsets of each sequence subgroup in flattened token batch.
	- used for chunked prompt, recompute, speculative multi-token scheduling.
	- note: vLLM often provides cumulative `subquery_lens` with same shape.

#### Block table inputs

- `[7] block_indices` (`i32`, `[num_blocks]`):
	- physical block IDs into cache tensors (0th dim indices).

- `[8] block_indices_begins` (`i32`, `[batch_size_in_sequences + 1]`):
	- start offset in `block_indices` per sequence.

Rules:
- blocks of each sequence are contiguous inside `block_indices` (no gaps per sequence segment).
- includes both past-filled and newly reserved/partially-filled blocks.
- with eviction, block tables can differ across decoder layers.

#### Others 
- `[12] max_context_len`
    - Scalar `i32`, originally tied to attention matrix sizing.
    - Usually equals `max(past_lens + prompt_lens)`.
    - Can be derived by plugin from sequence metadata; candidate for eventual removal.
