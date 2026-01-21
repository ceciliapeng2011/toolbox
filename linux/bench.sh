#!/bin/bash
# echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# watch -n 1 'cat /sys/devices/pci0000:00/0000:00:02.0/tile0/gt0/freq0/act_freq'

model_dir=$HOME/llm_irs/WW35_llm-optimum_2025.3.0-19807-RC2/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT
prompts_dir=$HOME/3rdparty/x-attention/eval/efficiency/xattn-32768.jsonl

export OV_VERBOSE=0

benchmark_enableDQ() {
    # enable DQ
    # export OV_GPU_ALLOW_BYPASS_XATTN_EXEC=1
    # python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --load_config '{"KV_CACHE_PRECISION": "f16"}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": 100}}"  -pf $prompts_dir -n 3 --infer_count 1
    # python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --load_config '{"KV_CACHE_PRECISION": "i8", "KEY_CACHE_QUANT_MODE":"BY_TOKEN"}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": 100}}"  -pf $prompts_dir -n 3 --infer_count 1

    export OV_GPU_ALLOW_BYPASS_XATTN_EXEC=0
    python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --load_config '{"KV_CACHE_PRECISION": "f16"}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": 100}}"  -pf $prompts_dir -n 3 --infer_count 1
    python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --load_config '{"KV_CACHE_PRECISION": "i8", "KEY_CACHE_QUANT_MODE":"BY_TOKEN"}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": 100}}"  -pf $prompts_dir -n 3 --infer_count 1
}

benchmark_disableDQ() {
    # diable DQ
    # export OV_GPU_ALLOW_BYPASS_XATTN_EXEC=1
    # python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --load_config '{"KV_CACHE_PRECISION": "f16", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": 100}}"  -pf $prompts_dir -n 3 --infer_count 1
    # python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --load_config '{"KV_CACHE_PRECISION": "i8", "KEY_CACHE_QUANT_MODE":"BY_TOKEN", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": 100}}"  -pf $prompts_dir -n 3 --infer_count 1

    export OV_GPU_ALLOW_BYPASS_XATTN_EXEC=0
    python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --load_config '{"KV_CACHE_PRECISION": "f16", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": 100}}"  -pf $prompts_dir -n 3 --infer_count 1
    python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --load_config '{"KV_CACHE_PRECISION": "i8", "KEY_CACHE_QUANT_MODE":"BY_TOKEN", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": 100}}"  -pf $prompts_dir -n 3 --infer_count 1
}

profile() {
    export OV_GPU_ALLOW_BYPASS_XATTN_EXEC=0
    export CLI_InOrderQueue=1 
    export CLI_LogToFile=1
    export CLI_DevicePerformanceTiming=1
    cliloader -cdt --dump-dir ./dump.DQ.fp16.$1/ python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --load_config '{"KV_CACHE_PRECISION": "f16"}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": $1}}"  -pf $prompts_dir -n 3 --infer_count 1 > log.DQ.fp16.$1
    cliloader -cdt --dump-dir ./dump.DQ.i8.$1/ python $HOME/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU --load_config '{"KV_CACHE_PRECISION": "i8", "KEY_CACHE_QUANT_MODE":"BY_TOKEN"}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": $1}}"  -pf $prompts_dir -n 3 --infer_count 1 > log.DQ.i8.$1
}

profile 100
profile 0.99
profile 0.9
profile 0.1