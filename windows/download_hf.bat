
python -m pip install -r C:\ceciliapeng\openvino.genai\tools\llm_bench\requirements.txt

mkdir c:\multi_document_analyst
cd c:\multi_document_analyst
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-32B
cd Qwen3-32B
git lfs pull

cd ..
optimum-cli export openvino --model Qwen3-32B --task text-generation-with-past Qwen3-32B\ov\int8 --weight-format int8
optimum-cli export openvino --model Qwen3-32B --task text-generation-with-past Qwen3-32B\ov\int4 --weight-format int4