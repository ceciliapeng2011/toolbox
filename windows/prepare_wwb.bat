@REM # WWB
@REM https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/who_what_benchmark

cd openvino.genai\tools\who_what_benchmark\
pip install -e .

cd c:\ceciliapeng
pip install -r openvino.genai\tools\llm_bench\requirements.txt

pip uninstall openvino openvino-tokenizers openvino-genai -y
@REM pip install result/install_pkg/tools/openvino*

@REM 测试时设置的代理
@REM export HF_ENDPOINT=https://hf-mirror.com
@REM export https_proxy=http://proxy-dmz.intel.com:912


@REM HF_ENDPOINT=https://hf-mirror.com https_proxy=http://proxy-dmz.intel.com:912 OPENVINO_LOG_LEVEL=4 wwb --target-model /mnt/llm_irs/WW29_llm_optimum_3b7248ef/qwen2.5-vl-3b-instruct/pytorch/dldt/compressed_weights/OV_FP16-INT8_ASYM --model-type visual-text --genai --gt-data /home/ceciliapeng/wwb_test/accuracy/qwen2.5-vl-3b-instruct__pytorch/reference.csv --device GPU --output qwen2.5-vl-3b-instruct/INT8_ASYM/PR/ > qwen2.5-vl-3b-instruct.INT8_ASYM.PR.log 2>&1

@REM HF_ENDPOINT=https://hf-mirror.com https_proxy=http://proxy-dmz.intel.com:912 OPENVINO_LOG_LEVEL=4 wwb --target-model /mnt/llm_irs/WW29_llm_optimum_3b7248ef/qwen2.5-vl-3b-instruct/pytorch/dldt/FP16 --model-type visual-text --genai --gt-data /home/ceciliapeng/wwb_test/accuracy/qwen2.5-vl-3b-instruct__pytorch/reference.csv --device GPU --output qwen2.5-vl-3b-instruct/FP16/PR/ > qwen2.5-vl-3b-instruct.FP16.PR.log 2>&1

@REM unset CM_FE_DIR
@REM export CM_FE_DIR=/home/ceciliapeng/CM/
@REM <!-- likely dynamic_split_fuse false ok, not set or true bad -->
@REM HF_ENDPOINT=https://hf-mirror.com https_proxy=http://proxy-dmz.intel.com:912 OPENVINO_LOG_LEVEL=4 python /home/ceciliapeng/openvino.genai/tools/who_what_benchmark/whowhatbench/wwb.py --target-model /mnt/llm_irs/WW29_llm_optimum_3b7248ef/qwen2.5-vl-3b-instruct/pytorch/dldt/FP16 --model-type visual-text --genai --gt-data /home/ceciliapeng/wwb_test/dbg_reference.csv --device GPU --num-samples 11 --cb-config '{"dynamic_split_fuse": false}'