

#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# chmod +x run_wwb_grid.sh
# nohup bash ./run_wwb_grid.sh > run_wwb_grid.out 2>&1 &
# examples:
#   bash ./run_wwb_grid.sh                 # long-prompt OFF (default)
#   bash ./run_wwb_grid.sh --long-prompt   # long-prompt ON
#   bash ./run_wwb_grid.sh --no-long-prompt

LONG_PROMPT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --long-prompt)
      LONG_PROMPT=true
      ;;
    --no-long-prompt)
      LONG_PROMPT=false
      ;;
    -h|--help)
      echo "Usage: $0 [--long-prompt|--no-long-prompt]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--long-prompt|--no-long-prompt]" >&2
      exit 2
      ;;
  esac
  shift
done


############################################
# User-provided lists (from your base script)
############################################
MODEL_DIRS=(
  "$HOME/llm_irs/WW05_llm-optimum_2026.0.0-20947/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT"
  "$HOME/llm_irs/WW05_llm-optimum_2026.0.0-20947/minicpm4-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT"
  "$HOME/llm_irs/WW05_llm-optimum_2026.0.0-20947/phi-3-mini-128k-instruct/pytorch/ov/OV_FP16-4BIT_DEFAULT"
)

if [[ "$LONG_PROMPT" == "true" ]]; then
  GT_DIR="$HOME/llm_irs/AC_llm/wwb_ref_gt_data_cache_long_prompts/"
else
  GT_DIR="$HOME/llm_irs/AC_llm/wwb_ref_gt_data_cache/"
fi

GT_FILES=(
  "$GT_DIR/qwen3-8b__NAT/reference.csv"
  "$GT_DIR/minicpm4-8b__NAT/reference.csv"
  "$GT_DIR/phi-3-mini-128k-instruct__NAT/reference.csv"
)

MODEL_TYPE=text

CB_CONFIGS=(
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":false}'
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":100,"xattention_block_size":128}}'
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":100,"xattention_block_size":256}}'
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":0.99,"xattention_block_size":128}}'
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":0.99,"xattention_block_size":256}}'
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":0.9,"xattention_block_size":128}}'
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":0.9,"xattention_block_size":256}}'
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":0.6,"xattention_block_size":128}}'
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":0.6,"xattention_block_size":256}}'
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":0.1,"xattention_block_size":128}}'
  '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":0.1,"xattention_block_size":256}}'
)

OV_CONFIGS=(
  '{"KV_CACHE_PRECISION":"i8","KEY_CACHE_QUANT_MODE":"BY_CHANNEL"}'
  '{"KV_CACHE_PRECISION":"i8","KEY_CACHE_QUANT_MODE":"BY_TOKEN"}'
  '{"KV_CACHE_PRECISION":"f16"}'
)

############################################
# Environment (as in your base script)
############################################
export HF_ENDPOINT=https://hf-mirror.com
export https_proxy=http://proxy-dmz.intel.com:912
export OPENVINO_LOG_LEVEL=4

############################################
# Output roots
############################################
TS="$(date +"%Y%m%d_%H%M%S")"

readarray -t _VERS < <(python - <<'PY'
def ver_openvino():
  try:
    import openvino
    return str(openvino.get_version())
  except Exception:
    return "unknown"

def ver_openvino_genai():
  try:
    import openvino_genai
    return str(openvino_genai.get_version())
  except Exception:
    return "unknown"

print(ver_openvino())
print(ver_openvino_genai())
PY
)

OV_VERSION="${_VERS[0]:-unknown}"
OV_GENAI_VERSION="${_VERS[1]:-unknown}"

# Make version strings path-safe
OV_VERSION_TAG="$(sed -E 's/[^A-Za-z0-9._-]+/-/g;s/-+/-/g;s/^-|-$//g' <<<"$OV_VERSION")"
OV_GENAI_VERSION_TAG="$(sed -E 's/[^A-Za-z0-9._-]+/-/g;s/-+/-/g;s/^-|-$//g' <<<"$OV_GENAI_VERSION")"

if [[ "$LONG_PROMPT" == "true" ]]; then
  LONG_PROMPT_TAG="lp-on"
else
  LONG_PROMPT_TAG="lp-off"
fi

LOG_ROOT="./logs.wwb_${TS}__${LONG_PROMPT_TAG}__ov-${OV_VERSION_TAG}__genai-${OV_GENAI_VERSION_TAG}"
OUTPUT_ROOT="./outputs.wwb_${TS}__${LONG_PROMPT_TAG}__ov-${OV_VERSION_TAG}__genai-${OV_GENAI_VERSION_TAG}"   # change if you want a different root

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

############################################
# Helpers
############################################

# Extract model name: the segment immediately before 'pytorch'
get_model_name_from_path() {
  local path="$1"
  awk -F'/' '
    {
      for (i=1; i<=NF; i++) if ($i=="pytorch") { print $(i-1); exit }
    }
  ' <<<"$path"
}

# Find the GT file containing the model name followed by "__"
find_gt_for_model() {
  local model_name="$1"
  local gt_match=""
  for gt in "${GT_FILES[@]}"; do
    if [[ "$gt" == *"/${model_name}__"* ]]; then
      gt_match="$gt"
      break
    fi
  done
  echo "$gt_match"
}

# Safe string for folder/file names (keep alnum, dash, underscore, dot)
safe_tag() {
  local s="$1"
  # replace spaces with nothing; map disallowed chars to dash
  s="${s// /}"
  s="$(sed -E 's/[^A-Za-z0-9._-]+/-/g;s/-+/-/g;s/^-|-$//g' <<<"$s")"
  echo "$s"
}

# -------- Parse OV_CONFIG to readable tag --------
# Expected keys (present in your configs):
#   KV_CACHE_PRECISION: i8/f16/...
#   KEY_CACHE_QUANT_MODE: BY_TOKEN / (optional)
ov_tag_from_json() {
  local json="$1"

  # Extract KV_CACHE_PRECISION
  local kvp
  kvp="$(sed -nE 's/.*"KV_CACHE_PRECISION"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' <<<"$json")"
  [[ -z "$kvp" ]] && kvp="na"

  # Extract KEY_CACHE_QUANT_MODE (optional)
  local kq
  kq="$(sed -nE 's/.*"KEY_CACHE_QUANT_MODE"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' <<<"$json")"

  local tag="ov_kv-${kvp}"
  if [[ -n "$kq" ]]; then
    # normalize BY_TOKEN -> by_token
    local kq_norm
    kq_norm="$(tr '[:upper:]' '[:lower:]' <<<"$kq")"
    tag="${tag}_kq-${kq_norm}"
  fi

  echo "$(safe_tag "$tag")"
}

# -------- Parse CB_CONFIG to readable tag --------
# We extract:
#   use_sparse_attention: true/false
#   sparse_attention_config.mode: XATTENTION (if present)
#   sparse_attention_config.xattention_threshold: number or fraction (e.g., 100, 0.9, 0.1)
#   sparse_attention_config.xattention_block_size: integer
#   max_num_batched_tokens: integer (kept to differentiate if needed)
cb_tag_from_json() {
  local json="$1"

  local use_sparse
  use_sparse="$(sed -nE 's/.*"use_sparse_attention"[[:space:]]*:[[:space:]]*(true|false).*/\1/p' <<<"$json")"
  [[ -z "$use_sparse" ]] && use_sparse="false"

  local mtoks
  mtoks="$(sed -nE 's/.*"max_num_batched_tokens"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' <<<"$json")"

  if [[ "$use_sparse" == "true" ]]; then
    local mode thr bsz
    mode="$(sed -nE 's/.*"mode"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' <<<"$json")"
    thr="$(sed -nE 's/.*"xattention_threshold"[[:space:]]*:[[:space:]]*([-0-9.]+).*/\1/p' <<<"$json")"
    bsz="$(sed -nE 's/.*"xattention_block_size"[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' <<<"$json")"

    [[ -z "$mode" ]] && mode="unknown"
    [[ -z "$thr" ]] && thr="na"
    [[ -z "$bsz" ]] && bsz="na"

    # normalize mode to lowercase (e.g., XATTENTION -> xattention)
    local mode_norm
    mode_norm="$(tr '[:upper:]' '[:lower:]' <<<"$mode")"

    local tag="cb_sparse-${mode_norm}_thr-${thr}_bs-${bsz}"
    if [[ -n "$mtoks" ]]; then
      tag="${tag}_mtoks-${mtoks}"
    fi
    echo "$(safe_tag "$tag")"
  else
    local tag="cb_dense"
    if [[ -n "$mtoks" ]]; then
      tag="${tag}_mtoks-${mtoks}"
    fi
    echo "$(safe_tag "$tag")"
  fi
}

# Return success (0) when xattention_threshold exists and is < 1.0, otherwise 1.
cb_has_threshold_lt_one() {
  local json="$1"
  local thr
  thr="$(sed -nE 's/.*"xattention_threshold"[[:space:]]*:[[:space:]]*([-0-9.]+).*/\1/p' <<<"$json")"

  if [[ -z "$thr" ]]; then
    return 1
  fi

  awk -v t="$thr" 'BEGIN { exit !(t + 0 < 1.0) }'
}

############################################
# Main loop
############################################

RUN_TS="$(date +%Y%m%d_%H%M%S)"

echo "long_prompt=${LONG_PROMPT}"
echo "gt_dir=${GT_DIR}"
echo "openvino_version=${OV_VERSION}"
echo "openvino_genai_version=${OV_GENAI_VERSION}"

for model_dir in "${MODEL_DIRS[@]}"; do
  if [[ ! -d "$model_dir" ]]; then
    echo "⚠️  Skip: model_dir not found: $model_dir" >&2
    continue
  fi

  model_name="$(get_model_name_from_path "$model_dir")"
  if [[ -z "$model_name" ]]; then
    echo "⚠️  Skip: could not extract model_name from: $model_dir" >&2
    continue
  fi

  gt_file="$(find_gt_for_model "$model_name")"
  if [[ -z "$gt_file" ]]; then
    echo "⚠️  Skip: no GT file matched for model_name: $model_name" >&2
    echo "     Searched among:" >&2
    printf '       - %s\n' "${GT_FILES[@]}" >&2
    continue
  fi

  if [[ ! -f "$gt_file" ]]; then
    echo "⚠️  Skip: matched GT file does not exist: $gt_file" >&2
    continue
  fi

  # One level per model and timestamp to keep things tidy
  model_root="${OUTPUT_ROOT}/${model_name}/${RUN_TS}"
  log_root="${LOG_ROOT}/${model_name}/${RUN_TS}"
  mkdir -p "$model_root" "$log_root"

  for ov_config in "${OV_CONFIGS[@]}"; do
    ov_tag="$(ov_tag_from_json "$ov_config")"

    for cb_config in "${CB_CONFIGS[@]}"; do
      if [[ "$LONG_PROMPT" != "true" ]] && cb_has_threshold_lt_one "$cb_config"; then
        echo "⏭️  Skip CB config (xattention_threshold < 1.0 while long-prompt is OFF)"
        continue
      fi

      cb_tag="$(cb_tag_from_json "$cb_config")"

      # Output dir & log name that clearly identify model / ov / cb
      out_dir="${model_root}/${ov_tag}/${cb_tag}"
      mkdir -p "$out_dir"

      out_log="${log_root}/wwb_${ov_tag}__${cb_tag}.log"

      echo "▶️  Running:"
      echo "    model_dir = $model_dir"
      echo "    model     = $model_name"
      echo "    gt_file   = $gt_file"
      echo "    ov_tag    = $ov_tag"
      echo "    cb_tag    = $cb_tag"
      echo "    output    = $out_dir"
      echo "    log       = $out_log"

      # Save exact configs for full traceability
      printf '%s\n' "$ov_config" > "${out_dir}/ov_config.json"
      printf '%s\n' "$cb_config" > "${out_dir}/cb_config.json"

      # Execute
      cmd=(
        wwb
        --target-model "$model_dir"
        --model-type "$MODEL_TYPE"
        --genai
        --gt-data "$gt_file"
        --device GPU
        --cb-config "$cb_config"
        --ov-config "$ov_config"
        --output "$out_dir"
      )

      if [[ "$LONG_PROMPT" == "true" ]]; then
        cmd+=(--long-prompt)
      fi

      "${cmd[@]}" >"$out_log" 2>&1

      echo "✅ Done: $model_name | $ov_tag | $cb_tag"
      echo
    done
  done
done

echo "🎉 All runs completed."
echo "    Outputs: $OUTPUT_ROOT"
echo "    Logs   : $LOG_ROOT"
