@REM call C:\ceciliapeng\openvino.venv\Scripts\activate

set PYTHONPATH=C:\ceciliapeng\openvino\release_install\python;C:\ceciliapeng\openvino\tools\ovc;C:\ceciliapeng\openvino\bin\intel64\Release\python
set OPENVINO_LIB_PATHS=C:\ceciliapeng\openvino\release_install\runtime\bin\intel64\Release\;C:\ceciliapeng\openvino\temp\Windows_AMD64\tbb\bin;C:\ceciliapeng\openvino\bin\intel64\Release\
set PATH=C:\ceciliapeng\openvino\tools\ovc;C:\ceciliapeng\openvino\bin\intel64\Release;C:\ceciliapeng\openvino\temp\Windows_AMD64\tbb\bin;%PATH%

set CM_FE_DIR=c:\ceciliapeng\ComputeSDK_Windows_internal_2025_WW41\compiler\bin

@REM set no_proxy=localhost,127.0.0.0/8,::1
@REM set ftp_proxy=http://child-prc.intel.com:913/
@REM set https_proxy=http://child-prc.intel.com:913/
@REM set http_proxy=http://child-prc.intel.com:913/
@REM set HF_ENDPOINT=https://hf-mirror.com

@REM DO NOT add any "" for these ENVs
@REM mkdir -p dump_debug_text
set OV_GPU_DUMP_TENSORS_FORMAT=binary
set OV_GPU_DUMP_TENSORS=all
set OV_GPU_DUMP_ITERATIONS=0
set OV_GPU_DUMP_SRC_TENSORS_AFTER_EXEC=1

@REM set OV_GPU_DUMP_SOURCES_PATH=micro_pa_cl_sources\
set NEO_CACHE_PERSISTENT=0


@REM int4
@REM set OV_GPU_DUMP_TENSORS_PATH=C:\ceciliapeng\dump_debug_bin_int4_PagedAttentionExtension_40206\
@REM set OV_GPU_DUMP_LAYER_NAMES=PagedAttentionExtension_40206
@REM set OV_GPU_DUMP_LAYER_NAMES=PagedAttentionExtension_40206 PagedAttentionExtension_40243 PagedAttentionExtension_40280 PagedAttentionExtension_40317 PagedAttentionExtension_40354 PagedAttentionExtension_40391 PagedAttentionExtension_40428 PagedAttentionExtension_40465 PagedAttentionExtension_40502 PagedAttentionExtension_40539 PagedAttentionExtension_40576 PagedAttentionExtension_40613 PagedAttentionExtension_40650 PagedAttentionExtension_40687 PagedAttentionExtension_40724 PagedAttentionExtension_40761 PagedAttentionExtension_40798 PagedAttentionExtension_40835 PagedAttentionExtension_40872 PagedAttentionExtension_40909 PagedAttentionExtension_40946 PagedAttentionExtension_40983 PagedAttentionExtension_41020 PagedAttentionExtension_41057 PagedAttentionExtension_41094 PagedAttentionExtension_41131 PagedAttentionExtension_41168 PagedAttentionExtension_41205 PagedAttentionExtension_41242 PagedAttentionExtension_41279 PagedAttentionExtension_41316 PagedAttentionExtension_41353 PagedAttentionExtension_41390 PagedAttentionExtension_41427 PagedAttentionExtension_41464 PagedAttentionExtension_41501 PagedAttentionExtension_41538 PagedAttentionExtension_41575 PagedAttentionExtension_41612 PagedAttentionExtension_41649 PagedAttentionExtension_41686 PagedAttentionExtension_41723 PagedAttentionExtension_41760 PagedAttentionExtension_41797 PagedAttentionExtension_41834 PagedAttentionExtension_41871 PagedAttentionExtension_41908 PagedAttentionExtension_41945

@REM int8, problem starting with layer9 PagedAttentionExtension_38747
@REM set OV_GPU_DUMP_TENSORS_PATH=C:\ceciliapeng\dump_debug_bin_int8_PagedAttentionExtension_38414\
set OV_GPU_DUMP_LAYER_NAMES=PagedAttentionExtension_38414
@REM set OV_GPU_DUMP_LAYER_NAMES=PagedAttentionExtension_38414 PagedAttentionExtension_38451 PagedAttentionExtension_38488 PagedAttentionExtension_38525 PagedAttentionExtension_38562 PagedAttentionExtension_38599 PagedAttentionExtension_38636 PagedAttentionExtension_38673 PagedAttentionExtension_38710 PagedAttentionExtension_38747 PagedAttentionExtension_38784 PagedAttentionExtension_38821 PagedAttentionExtension_38858 PagedAttentionExtension_38895 PagedAttentionExtension_38932 PagedAttentionExtension_38969 PagedAttentionExtension_39006 PagedAttentionExtension_39043 PagedAttentionExtension_39080 PagedAttentionExtension_39117 PagedAttentionExtension_39154 PagedAttentionExtension_39191 PagedAttentionExtension_39228 PagedAttentionExtension_39265 PagedAttentionExtension_39302 PagedAttentionExtension_39339 PagedAttentionExtension_39376 PagedAttentionExtension_39413 PagedAttentionExtension_39450 PagedAttentionExtension_39487 PagedAttentionExtension_39524 PagedAttentionExtension_39561 PagedAttentionExtension_39598 PagedAttentionExtension_39635 PagedAttentionExtension_39672 PagedAttentionExtension_39709 PagedAttentionExtension_39746 PagedAttentionExtension_39783 PagedAttentionExtension_39820 PagedAttentionExtension_39857 PagedAttentionExtension_39894 PagedAttentionExtension_39931 PagedAttentionExtension_39968 PagedAttentionExtension_40005 PagedAttentionExtension_40042 PagedAttentionExtension_40079 PagedAttentionExtension_40116 PagedAttentionExtension_40153


set OV_VERBOSE=1
@REM set ONEDNN_VERBOSE=s
set OPENVINO_LOG_LEVEL=0

set model_dir=c:\multi_document_analyst\Qwen3-32B\ov\int8
@REM set prompt_dir=C:\Users\Local_Admin\peter\frameworks.ai.openvino.llm.prompts\2048\qwen3-30b-a3b.jsonl

@REM set cb_config="{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":true, \"sparse_attention_config\":{\"mode\": \"XATTENTION\", \"xattention_threshold\": 100}}"
set cb_config="{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096, \"use_sparse_attention\":false}"
set lc_config="{\"KV_CACHE_PRECISION\":\"i8\", \"KEY_CACHE_QUANT_MODE\":\"BY_TOKEN\"}"
set bench=C:\ceciliapeng\openvino.genai\tools\llm_bench\benchmark.py
set py=C:\ceciliapeng\openvino.venv\Scripts\python.exe

%py% --version
@REM %py% %bench% -m %model_dir% -pf "xatten-1265.jsonl" --disable_prompt_permutation -d GPU --load_config %lc_config% --cb_config %cb_config%  -n 0 --infer_count 1 --apply_chat_template > log.llm_bench
%py% %bench% -m %model_dir% -pf "xatten-248.jsonl" --disable_prompt_permutation -d GPU --load_config %lc_config% --cb_config %cb_config%  -n 0 --infer_count 1 --apply_chat_template > log.temp

@REM "C:\Program Files (x86)\Intel\oneAPI\vtune\2025.6\bin64\vtune" -collect gpu-hotspots -knob gpu-sampling-interval=0.1 -data-limit=10000 -r C:\Users\Local_Admin\river\vtune_logs\qwen3_moe_30B_2 --app-working-dir=C:\ceciliapeng\openvino.genai\tools\llm_bench -- C:\ceciliapeng\openvino.genai\tools\llm_bench\run_llm.bat

