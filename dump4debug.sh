#!/usr/bin/env bash
set -euo pipefail

export PATH=$PATH:$HOME/OCL/opencl-intercept-layer/build_RelWithDebInfo/install/bin:$HOME/OCL/opencl-intercept-layer/build_Release/install/bin/
SDPA_COUNT=36


run_llm_app() {
    model_dir=$HOME/llm_irs/WW02_llm-optimum_2026.0.0-20769/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT
    # model_dir=$HOME/llm_irs/WW02_llm-optimum_2026.0.0-20769/phi-3-mini-128k-instruct/pytorch/ov/OV_FP16-4BIT_DEFAULT
    # model_dir=$HOME/llm_irs/WW02_llm-optimum_2026.0.0-20769/minicpm4-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT
    # Input token size: 4477 
    # prompts_dir=$HOME/3rdparty/x-attention/eval/efficiency/xattn-32768.jsonl
    prompts_dir=$HOME/3rdparty/x-attention/eval/efficiency/xattn-4096.jsonl
    # Input token size: 1125
    # prompts_dir=$HOME/3rdparty/x-attention/eval/efficiency/xattn-1024.jsonl
    # prompts_dir=$HOME/3rdparty/x-attention/eval/efficiency/xattn-102.jsonl
    # cb_config='{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":false}'
    cb_config='{"enable_prefix_caching":false,"max_num_batched_tokens":4096,"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":0.9,"xattention_block_size":256}}'
    # load_config='{"KV_CACHE_PRECISION":"i8","KEY_CACHE_QUANT_MODE":"BY_TOKEN"}'
    load_config='{"KV_CACHE_PRECISION":"f16"}'


    # python $HOME/openvino/large_context.py --use-sparse-xattention --xattn-block-size=256

    python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --cb_config $cb_config --load_config $load_config -pf $prompts_dir -n 0 --infer_count 10

    # gt_file="$HOME/llm_irs/AC_llm/wwb_ref_gt_data_cache/2026.0.0-20769-87b915269ed_nat_ref/CPU_ICX/default_data_wwb_long_prompt/cache_nat_refs_cli___long_prompt/qwen3-8b__NAT/reference.csv"
    # gt_file="reference.csv"
    # out_dir=./tmp.wwb/
    # mkdir -p "${out_dir}"
    # wwb \
    # --target-model "$model_dir" \
    # --model-type "text" \
    # --long-prompt \
    # --genai \
    # --gt-data "$gt_file" \
    # --device GPU \
    # --cb-config "$cb_config" \
    # --ov-config "$load_config" \
    # --output "$out_dir"
}
run_llm_app

run_one() {
    mkdir -p dump_debug_xattn
    export DUMP_XATTN_INTERNALS=./dump_debug_xattn/
    mkdir -p dump_debug_bin_PagedAttentionExtension_27964
    export OV_GPU_DUMP_TENSORS_PATH=./dump_debug_bin_PagedAttentionExtension_27964/
    export OV_GPU_DUMP_TENSORS_FORMAT=binary
    export OV_GPU_DUMP_TENSORS=all
    export OV_GPU_DUMP_ITERATIONS="0"
    export OV_GPU_DUMP_SRC_TENSORS_AFTER_EXEC=1
    export OV_GPU_DUMP_LAYER_NAMES=PagedAttentionExtension_27964
    run_llm_app > log 2>&1
}
# run_one


run_all() {
    # first round to get all PA layers.
    # CLI_InOrderQueue=1 CLI_CallLogging=1 CLI_LogToFile=1 CLI_FinishAfterEnqueue=1 cliloader \
    OV_VERBOSE=4 run_llm_app > tmp.LOG 2>&1

    grep "Execute pagedattentionextension:PagedAttentionExtension_" tmp.LOG |awk -F ':' '{print $6}' |awk -F ' ' '{print $1}'|head -n $SDPA_COUNT | paste -sd ' ' > tmp.pa.log
    export OV_GPU_DUMP_LAYER_NAMES="$(cat tmp.pa.log)"
    echo $OV_GPU_DUMP_LAYER_NAMES

    # # second round to dump all PA layers' inputs/outputs in text
    mkdir -p dump_debug_xattn1
    export DUMP_XATTN_INTERNALS=./dump_debug_xattn1/
    mkdir -p dump_debug_text
    export OV_GPU_DUMP_TENSORS_PATH=./dump_debug_text/
    export OV_GPU_DUMP_TENSORS_FORMAT=text
    export OV_GPU_DUMP_TENSORS=all
    export OV_GPU_DUMP_ITERATIONS="0"
    export OV_GPU_DUMP_SRC_TENSORS_AFTER_EXEC=1
    run_llm_app > tmp.txt.LOG 2>&1

    # # third round to dump all PA layers' inputs/outputs in binary
    mkdir -p dump_debug_xattn2
    export DUMP_XATTN_INTERNALS=./dump_debug_xattn2/
    mkdir -p dump_debug_binary
    export OV_GPU_DUMP_TENSORS_PATH=./dump_debug_binary/
    export OV_GPU_DUMP_TENSORS_FORMAT=binary
    export OV_GPU_DUMP_TENSORS=all
    export OV_GPU_DUMP_ITERATIONS="0"
    export OV_GPU_DUMP_SRC_TENSORS_AFTER_EXEC=1
    run_llm_app > tmp.bin.LOG 2>&1
}
# run_all

