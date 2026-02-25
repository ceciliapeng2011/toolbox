#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_llm_app                     # uses defaults threshold=0.99, block=256
#   run_llm_app -t 0.97 -b 512      # custom threshold and block size
#   run_llm_app --threshold 0.95 --block-size 128
#
# Notes:
# - Directory name will embed the chosen values, e.g.:
#   ./report.clintercept.i8.block256.xattn0.99/
#
# - Assumes: $model_dir and $prompts_dir are set in the environment.

model_dir=$HOME/llm_irs/WW02_llm-optimum_2026.0.0-20769/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT
prompts_dir=$HOME/3rdparty/x-attention/eval/efficiency/xattn-32768.jsonl

run_llm_app() {
    # Defaults
    local xattention_threshold="0.99"
    local xattention_block_size="256"

    # Parse args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -t|--threshold)
                xattention_threshold="${2:?Missing value for --threshold}"
                shift 2
                ;;
            -b|--block-size)
                xattention_block_size="${2:?Missing value for --block-size}"
                shift 2
                ;;
            -h|--help)
                echo "Usage: run_llm_app [-t|--threshold FLOAT] [-b|--block-size INT]"
                return 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                echo "Try: run_llm_app -h" >&2
                return 2
                ;;
        esac
    done

    # Build dump dir that reflects the args
    local dump_dir="./report.clintercept.i8.block${xattention_block_size}.xattn${xattention_threshold}"

    # JSON blobs (escaped for shell)
    local cb_config
    cb_config=$(cat <<EOF
{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":${xattention_threshold},"xattention_block_size":${xattention_block_size}}}
EOF
)

    local load_config
    load_config=$(cat <<'EOF'
{"KV_CACHE_PRECISION":"i8","KEY_CACHE_QUANT_MODE":"BY_TOKEN"}
EOF
)

    # Environment flags for cliloader
    CLI_InOrderQueue=1 \
    CLI_LogToFile=1 \
    CLI_DevicePerformanceTiming=1 \
    cliloader --dump-dir "${dump_dir}/" \
    python "$HOME/openvino.genai/tools/llm_bench/benchmark.py" \
        -m "$model_dir" \
        --disable_prompt_permutation \
        -d GPU \
        --cb_config "$cb_config" \
        --load_config "$load_config" \
        -pf "$prompts_dir" \
        -n 0 \
        --infer_count 1

    # Post-process the CLIntercept report
    python3 "$HOME/OCL/libraries.ai.videoanalyticssuite.gpu-tools/format_clintercept.py" \
        "${dump_dir}/clintercept_report.txt"
}

OV_GPU_ALLOW_BYPASS_XATTN_EXEC=0 run_llm_app -t 100 -b 256 > log.256_100.xattn 2>&1
OV_GPU_ALLOW_BYPASS_XATTN_EXEC=1 run_llm_app -t 100 -b 256 > log.256_100 2>&1
run_llm_app -t 0.99 -b 256 > log.256_0.99 2>&1
run_llm_app -t 0.9 -b 256 > log.256_0.9 2>&1
run_llm_app -t 0.6 -b 256 > log.256_0.6 2>&1

OV_GPU_ALLOW_BYPASS_XATTN_EXEC=0 run_llm_app -t 100 -b 128 > log.128_100.xattn 2>&1
OV_GPU_ALLOW_BYPASS_XATTN_EXEC=1 run_llm_app -t 100 -b 128 > log.128_100 2>&1
run_llm_app -t 0.99 -b 128 > log.128_0.99 2>&1
run_llm_app -t 0.9 -b 128 > log.128_0.9 2>&1
run_llm_app -t 0.6 -b 128 > log.128_0.6 2>&1
