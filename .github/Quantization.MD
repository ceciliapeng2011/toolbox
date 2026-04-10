# OpenVINO K/V cache quantization strategies and layout

## TASK1. Learn K/V cache data layout format of PagedAttention under each quantizations from openvino code base:

1. GPU Plugin OCL path - 
Read $HOME/openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/pa_kv_cache_update_ref.cl

2. GPU Plugin CM path -
Read $HOME/openvino/src/plugins/intel_gpu/src/graph/impls/cm/pa_kv_cache_update_ref.cm

2. CPU Plugin path -
Read $HOME/workspace/openvino/src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/attn_quant.hpp
Read $HOME/workspace/openvino/src/plugins/intel_cpu/src/nodes/scaled_attn.cpp:2172
Read $HOME/workspace/openvino/src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/attn_quant.cpp


Ignore Turboquant, SageAttention.

Summarize your learnings at end of this file.


## Summary

### Common paged-attention placement

- Logical cache tensors are key_cache[num_blocks, kv_heads, block_size, k_head_size] and value_cache[num_blocks, kv_heads, block_size, v_head_size].
- Block and token position are computed from past_lens + subsequence offsets and block index tables.

### GPU OCL path (pa_kv_cache_update_ref.cl)

- Non-quantized layout:
	- Key write uses hidden-major order inside a block: base + token_pos + hidden * block_size.
	- Value write uses token-major order inside a block: base + token_pos * v_head_size + hidden.
- Per-token quantization (quantize_and_save_per_token):
	- Computes min/max for one token over each quantization group.
	- Stores payload as u8 or packed u4.
	- Stores metadata in a block tail as two token-length arrays:
		- scale_inv[token_pos] = 1 / scale
		- zp[token_pos] = zp
- Key by-channel mode (IS_KEY_BY_CHANNEL):
	- Key cache uses by-channel quantization with adjusted block stride.
	- On partially filled block, previously quantized key data is dequantized, merged with new tokens, then requantized.
	- Value remains per-token quantized.

### GPU CM path (pa_kv_cache_update_ref.cm)

- With KV_CACHE_COMPRESSION_PER_TOKEN, payload type is u8.
- For each token, kernel quantizes across full head and writes to out + out_offset + token_pos * HEAD_SIZE.
- Metadata is appended after payload region as half arrays for scale_inv and zp (indexed by token position).

### CPU path (attn_quant.hpp / attn_quant.cpp / scaled_attn.cpp)

- Runtime config selects key quant mode (BY_CHANNEL or BY_TOKEN), and group size defaults to full head size if not set.
- By-token / by-dims path (quantize_block_by_dims) uses explicit interleaved group layout:
	- [scale0, zp0] | [group0] | [scale1, zp1] | [group1] | ...
	- For u4, each group payload is packed (group_size / 2 bytes).
- By-channel path (quantize_block_by_channel) uses block-header metadata:
	- [scales[S]] [zps[S]] [quantized_data], where S is hidden size.
	- For mixed blocks, previous data is dequantized, new tokens are appended, and valid region is requantized.

### Key takeaways

- OCL non-quant K/V traversal is intentionally asymmetric (K hidden-major, V token-major).
- Metadata placement differs by backend, but all paths preserve scale/zp needed for append-time reconstruction.
- Mixed-block requantization is the core mechanism for by-channel correctness in both GPU and CPU paths.

## TASK2. Is Q8_0 and Q4_1 quantization of llama.cpp supported by any plugin for K/V cache?
Note: 1. Here is asking about K/V cache, not model weights. 2. Just care about the quantization algorithm and data layout; Name itself does not matter.

### TASK2 Findings

- **Q4_1-like (4-bit affine: scale + offset/zp): supported on all listed plugin paths.**
	- **GPU OCL path:** **Supported** (`u4` + scale/zp metadata, per-token or by-channel).
		- **Memory layout (per-token key/value path):** packed 4-bit payload in block + metadata tail.
		- Metadata tail is token-indexed arrays (scale inverse and zp); for key by-channel, data is channel-oriented with adjusted block stride and per-channel params.
	- **GPU CM path:** **Not shown as `u4` in this kernel** (this CM file is `u8` per-token compression path), so **Q4_1-like is not demonstrated here**.
		- **Memory layout in this file:** token-major `u8` payload + appended fp16 scale/zp arrays, not 4-bit packed layout.
	- **CPU path:** **Supported** (`u4` path with affine params in grouped/block layouts).
		- **Memory layout (by-dims):** `[scale,zp] | packed-group-data | [scale,zp] | packed-group-data | ...`.
		- **Memory layout (by-channel):** `[scales[S]] [zps[S]] [quantized_data]` per block.
	- **Why:** Q4_1-like means affine low-bit quant with stored scale/offset; OCL+CPU K/V cache implement that class, though memory packing differs from llama.cpp GGML.


- **Q8_0-like (int8 symmetric, scale-only): not supported in the listed paged-attention plugin paths.**
	- **GPU OCL path:** **Not Q8_0-like** for K/V quantized path; it is mostly affine (`u8` with zp) rather than symmetric scale-only int8.
		- **Memory layout evidence:** quantized bytes plus stored zp metadata (token-indexed tails / channel params), which violates scale-only requirement.
	- **GPU CM path:** **Not Q8_0-like** in this file; per-token `u8` stores both scale and zp.
		- **Memory layout evidence:** payload region followed by fp16 `scale_inv[]` and `zp[]` arrays for each token.
	- **CPU path:** **Not Q8_0-like** for paged K/V quantized paths; they are affine (`u8`/`u4`, scale+zp).
		- **Memory layout evidence:** both CPU layouts explicitly allocate zp (`[scale,zp]` per group or `[zps[S]]` block header).
	- **Why:** Q8_0-like requires symmetric int8 with scale-only semantics, while these paths store/use zero-point (affine quantization).

- **Final answer:**
	- `Q4_1`-like: **supported** (OCL + CPU in studied paths; CM file here does not show `u4`).
	- `Q8_0`-like: **not supported** in the listed paged-attention plugin paths.
	- Exact llama.cpp GGML on-memory format is still **not** matched by plugin K/V cache layout.

## TASK2 Summary Table (Algo + Memory Layout + Matching)

| Quant target | Path | Quantization algorithm | Data layout in memory (conceptual) | Same algo as llama.cpp target? | Exact bit/layout match? | Comments |
|---|---|---|---|---|---|---|
| Q4_1 | llama.cpp / GGML reference | 4-bit affine per block (`q = round(x/d + m/d)` style, with scale+bias terms) | `block_q4_1 = [d(fp16), m(fp16), qs(packed nibbles)]` | Baseline | Baseline | Canonical definition used for comparison. |
| Q4_1 | GPU OCL paged-attn K/V | 4-bit affine (`u4`) with scale/zp metadata (per-token or by-channel modes) | Per-token mode: `[packed-u4 payload for block] + [scale_inv[token]] + [zp[token]]`; by-channel key mode stores channel-oriented params/data with adjusted stride | Yes, Q4_1-like | No | Same affine family, different granularity and metadata placement from GGML blocks. |
| Q4_1 | GPU CM paged-attn K/V (this file) | `u8` affine per-token (not 4-bit) | `[u8 token-major payload] + [scale_inv[token] fp16 array] + [zp[token] fp16 array]` | No (for Q4_1) | No | This CM kernel demonstrates u8 compression, not q4 packing. |
| Q4_1 | CPU paged-attn K/V | 4-bit affine (`u4`) with scale/zp | `[scale,zp] + packed-group-data + [scale,zp] + packed-group-data + ...` | Yes, Q4_1-like | No | This row uses the by-dims view; algorithm matches Q4_1 family, but exact GGML block layout is different. |
| Q8_0 | llama.cpp / GGML reference | 8-bit symmetric per block, scale-only (no zp) | `block_q8_0 = [d(fp16), qs(int8[32])]` | Baseline | Baseline | Canonical scale-only int8 block format. |
| Q8_0 | GPU OCL paged-attn K/V | 8-bit affine (`u8`, with zp) | Quantized bytes plus stored zp/scale metadata (token tails or channel params depending mode) | No | No | Uses affine zero-point path, not symmetric scale-only q8_0 style. |
| Q8_0 | GPU CM paged-attn K/V (this file) | 8-bit affine per-token (`u8` + zp) | `[u8 token-major payload] + [scale_inv[token]] + [zp[token]]` | No | No | Explicit zp array means not q8_0 algorithm/layout. |
| Q8_0 | CPU paged-attn K/V | Affine `u8`/`u4` with zp for listed paged-attn paths | By-dims uses repeated `[scale,zp]` headers; by-channel uses `[scales[S]] [zps[S]]` header | No | No | For this task scope (excluding SageAttention), no scale-only int8 q8_0-equivalent path. |


## Take-aways
1. Q8_0 is not supported by any OV path, as it is symmetric quantization, while OV uses affine quantization.
2. Q4_1 is conceptually supported by OV OCL path and OV CPU plugin, but the data layout is different. The most close case is by_token quantization with group size 32 in CPU plugin, but CPU plugin uses float (4bytes) for scale/zp, while llama.cpp uses fp16.
3. BTW, opengino.genai uses gguf_utils to support both formats, but it is for weights, not KV cache.