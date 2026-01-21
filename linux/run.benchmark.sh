
#!/usr/bin/env bash
# Benchmark all model_dir x prompts_dir x cb_config variants x KV precision combinations.
# - Adds XATTENTION runs over thresholds {100, 0.99, 0.9, 0.6, 0.1} and block sizes {128, 256}
# - Adds dense runs (use_sparse_attention=false)
# - Safe JSON quoting and log naming
# - Timestamped logs in a single folder
# - Sequential execution (toggleable if needed)

set -Eeuo pipefail
IFS=$'\n\t'

# echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# watch -n 1 'cat /sys/devices/pci0000:00/0000:00:02.0/tile0/gt0/freq0/act_freq'

# chmod +x run.benchmark.sh
# nohup bash ./run.benchmark.sh > run.benchmark.out 2>&1

########################################
# Config: edit these as needed
########################################

MODEL_DIRS=(
  "$HOME/llm_irs/WW02_llm-optimum_2026.0.0-20769/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT"
  "$HOME/llm_irs/WW02_llm-optimum_2026.0.0-20769/minicpm4-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT"
  "$HOME/llm_irs/WW02_llm-optimum_2026.0.0-20769/phi-3-mini-128k-instruct/pytorch/ov/OV_FP16-4BIT_DEFAULT"
)

PROMPT_FILES=(
  "$HOME/3rdparty/x-attention/eval/efficiency/xattn-102.jsonl"
  "$HOME/3rdparty/x-attention/eval/efficiency/xattn-1265.jsonl"
  "$HOME/3rdparty/x-attention/eval/efficiency/xattn-4096.jsonl"
  "$HOME/3rdparty/x-attention/eval/efficiency/xattn-32768.jsonl"
)

# Thresholds to sweep for XATTENTION mode
XATTN_THRESHOLDS=(100 0.99 0.9 0.6 0.1)

# Block sizes to sweep
XATTN_BLOCK_SIZES=(128 256)

# Dense (no sparse attention) cb_config
CB_CONFIG_DENSE='{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":false}'

# Benchmark script and device
BENCH_PY="$HOME/openvino.genai/tools/llm_bench/benchmark.py"
DEVICE="GPU"

# Prompt options
N_PROMPTS=1
INFER_COUNT=10

########################################
# Derived variables
########################################
TS="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="./logs.benchmark_${TS}"
mkdir -p "${LOG_DIR}"

# Load configs (KV cache precision variants)
LOAD_CONFIG_FP16='{"KV_CACHE_PRECISION":"f16"}'
LOAD_CONFIG_I8='{"KV_CACHE_PRECISION":"i8","KEY_CACHE_QUANT_MODE":"BY_TOKEN"}'

########################################
# Helpers
########################################

# Create a filesystem-safe identifier from a path:
# replaces '/' with '__', spaces with '_', removes trailing '/'
sanitize_id() {
  local path="$1"
  path="${path%/}"
  local s="${path//\//__}"
  s="${s// /_}"
  echo "${s}"
}

# Short name helper: basename without extension (for prompt files)
basename_no_ext() {
  local path="$1"
  local base
  base="$(basename -- "$path")"
  echo "${base%.*}"
}

run_and_log() {
  local -n cmd_ref=$1
  local logfile=$2
  echo "[$(date +'%F %T')] Starting: ${cmd_ref[*]}"
  {
    echo "### Started at: $(date +'%F %T')"
    echo "### Command: ${cmd_ref[*]}"
    "${cmd_ref[@]}"
    echo "### Finished at: $(date +'%F %T')"
  } > "${logfile}" 2>&1
  echo "[$(date +'%F %T')] Done. Log: ${logfile}"
}

########################################
# Main loop
########################################
for model_dir in "${MODEL_DIRS[@]}"; do
  model_id="$(sanitize_id "${model_dir}")"

  for prompts_dir in "${PROMPT_FILES[@]}"; do
    prompt_id_full="$(sanitize_id "${prompts_dir}")"
    prompt_id_base="$(basename_no_ext "${prompts_dir}")"

    # Choose one; full path gives uniqueness across folders
    prompt_id="${prompt_id_full}"
    # prompt_id="${prompt_id_base}"

    # ----------------------------
    # 1) Dense baseline (no sparse)
    # ----------------------------
    for kv in "fp16" "i8"; do
      if [[ "${kv}" == "fp16" ]]; then
        load_cfg="${LOAD_CONFIG_FP16}"
      else
        load_cfg="${LOAD_CONFIG_I8}"
      fi

      cfg_id="dense"
      FP_LOG="${LOG_DIR}/cm_pa.${kv}.${cfg_id}.${model_id}.${prompt_id}.${TS}.log"

      CMD_DENSE=(
        python "${BENCH_PY}"
        -m "${model_dir}"
        --disable_prompt_permutation
        -d "${DEVICE}"
        --cb_config "${CB_CONFIG_DENSE}"
        --load_config "${load_cfg}"
        -pf "${prompts_dir}"
        -n "${N_PROMPTS}"
        --infer_count "${INFER_COUNT}"
      )
      run_and_log CMD_DENSE "${FP_LOG}"
    done

    # ---------------------------------------------------------
    # 2) Sparse XATTENTION sweep over thresholds and block size
    # ---------------------------------------------------------
    for th in "${XATTN_THRESHOLDS[@]}"; do
      for bs in "${XATTN_BLOCK_SIZES[@]}"; do
        # Build cb_config with the current threshold and block size (JSON numeric)
        CB_CONFIG_XATTN=$(printf \
          '{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":%s,"xattention_block_size":%s}}' \
          "$th" "$bs")

        for kv in "fp16" "i8"; do
          if [[ "${kv}" == "fp16" ]]; then
            load_cfg="${LOAD_CONFIG_FP16}"
          else
            load_cfg="${LOAD_CONFIG_I8}"
          fi

          cfg_id="xattn-${th}-bs${bs}"
          LOG_FILE="${LOG_DIR}/cm_pa.${kv}.${cfg_id}.${model_id}.${prompt_id}.${TS}.log"

          CMD_XATTN=(
            python "${BENCH_PY}"
            -m "${model_dir}"
            --disable_prompt_permutation
            -d "${DEVICE}"
            --cb_config "${CB_CONFIG_XATTN}"
            --load_config "${load_cfg}"
            -pf "${prompts_dir}"
            -n "${N_PROMPTS}"
            --infer_count "${INFER_COUNT}"
          )
          run_and_log CMD_XATTN "${LOG_FILE}"
        done
      done
    done

  done
done

echo "All runs completed."
echo "Logs saved to: ${LOG_DIR}"
