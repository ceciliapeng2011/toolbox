#!/usr/bin/env bash

# MODEL_CACHE_URL="https://ov-share-13.iotg.sclab.intel.com/cv_bench_cache/WW02_llm-optimum_2026.0.0-20769"
# GT_CACHE_URL="https://ov-share-04.sclab.intel.com/cv_bench_cache/AC_llm/wwb_ref_gt_data_cache/2026.0.0-20769-87b915269ed_nat_ref/CPU_ICX/default_data_wwb_long_prompt/cache_nat_refs_cli___long_prompt/"

# MODEL_DIRS=(
#   "qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT"
#   "minicpm4-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT"
#   "phi-3-mini-128k-instruct/pytorch/ov/OV_FP16-4BIT_DEFAULT"
# )

# REF_DIRS=(
#     "qwen3-8b__NAT"
#     "minicpm4-8b__NAT"
#     "phi-3-mini-128k-instruct__NAT"
# )

# nohup wget -r -l 0 -nH --cut-dirs=1  --no-parent --reject="index.html*" --no-check-certificate $MODEL_CACHE_URL/$MODEL_DIRS
# nohup wget -r -l 0 -nH --cut-dirs=1  --no-parent --reject="index.html*" --no-check-certificate $GT_CACHE_URL/$REF_DIRS

#!/usr/bin/env bash
# Download multiple model and GT directories from base URLs using wget,
# with proxies explicitly disabled.
# - Iterates over arrays MODEL_DIRS and REF_DIRS
# - Logs each download separately
# - Supports sequential or background execution
# - Disables proxy via env and wget --no-proxy
set -euo pipefail

########################################
# Config
########################################
MODEL_CACHE_URL="https://ov-share-13.iotg.sclab.intel.com/cv_bench_cache/WW02_llm-optimum_2026.0.0-20769"
GT_CACHE_URL="https://ov-share-04.sclab.intel.com/cv_bench_cache/AC_llm/wwb_ref_gt_data_cache/2026.0.0-20769-87b915269ed_nat_ref/CPU_ICX/default_data_wwb_long_prompt/cache_nat_refs_cli___long_prompt/"

MODEL_DIRS=(
  "qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT"
  "minicpm4-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT"
  "phi-3-mini-128k-instruct/pytorch/ov/OV_FP16-4BIT_DEFAULT"
)

REF_DIRS=(
  "qwen3-8b__NAT"
  "minicpm4-8b__NAT"
  "phi-3-mini-128k-instruct__NAT"
)

# Where to store logs and downloads (customize if needed)
DOWNLOAD_ROOT="${PWD}/llm_irs"
LOG_ROOT="${PWD}/logs"

# Execution mode: set to "sequential" or "background"
EXEC_MODE="sequential"  # change to "background" to run in parallel

########################################
# Setup
########################################
mkdir -p "${DOWNLOAD_ROOT}" "${LOG_ROOT}"

# Timestamp
TS="$(date +"%Y%m%d_%H%M%S")"

# Verify wget is available
if ! command -v wget >/dev/null 2>&1; then
  echo "[ERROR] wget not found. Install it and retry."
  exit 1
fi

########################################
# Disable proxies for this script
########################################
# Unset common proxy environment variables (both uppercase and lowercase)
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY
unset https_proxy http_proxy all_proxy no_proxy

# Also prevent wget from using proxies via its CLI flag
WGET_NO_PROXY_OPT="--no-proxy"

########################################
# Helpers
########################################

# Sanitize a string for use in a filename (replace path separators and spaces)
sanitize_id() {
  local s="${1%/}"
  s="${s//\//__}"   # '/' -> '__' to retain hierarchy hint
  s="${s// /_}"     # spaces -> '_'
  echo "${s}"
}

# Run a single wget download with logging
download_dir() {
  local base_url="$1"
  local rel_dir="$2"
  local category="$3"  # "model" or "gt"

  # Full URL (ensure trailing slash)
  local url="${base_url%/}/${rel_dir%/}/"

  # Output subfolder: use last segment of rel_dir
  local tail="${rel_dir##*/}"
  #  local out_dir="${DOWNLOAD_ROOT}/${category}/${tail}"
  local out_dir="${DOWNLOAD_ROOT}"

  # Log file name encodes category and sanitized rel_dir
  local rel_id
  rel_id="$(sanitize_id "${rel_dir}")"
  local log_file="${LOG_ROOT}/wget_${category}.${rel_id}.${TS}.log"

  mkdir -p "${out_dir}"

  echo "[INFO] Downloading (${category}):
    URL      : ${url}
    OUT DIR  : ${out_dir}
    LOG FILE : ${log_file}
    PROXY    : disabled
  "

  # Compose wget command with --no-proxy
  local cmd=(
    wget -r -l 0
    --no-parent
    -nH --cut-dirs=1
    --reject "index.html*"
    --no-check-certificate
    "${WGET_NO_PROXY_OPT}"
    --directory-prefix="${out_dir}"
    -o "${log_file}"
    "${url}"
  )

  if [[ "${EXEC_MODE}" == "background" ]]; then
    nohup "${cmd[@]}" >/dev/null 2>&1 &
    local pid=$!
    echo "[INFO] Started background wget PID=${pid} (${category}) url=${url}"
    echo "${pid}|${category}|${url}|${log_file}" >> "${LOG_ROOT}/pids.${TS}.txt"
  else
    "${cmd[@]}"
    echo "[DONE] (${category}) ${url}"
  fi
}

########################################
# Download loops
########################################

# Models
for rel in "${MODEL_DIRS[@]}"; do
  download_dir "${MODEL_CACHE_URL}" "${rel}" "model"
done

# GT refs
for rel in "${REF_DIRS[@]}"; do
  download_dir "${GT_CACHE_URL}" "${rel}" "gt"
done

# If background mode, you can optionally wait for PIDs recorded in logs/pids.<TS>.txt
if [[ "${EXEC_MODE}" == "background" ]]; then
  echo "[INFO] Background mode enabled. Jobs started. Monitor logs under: ${LOG_ROOT}"
fi

echo "[ALL COMPLETED] Downloads root: ${DOWNLOAD_ROOT}"
