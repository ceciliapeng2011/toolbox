#!/usr/bin/env bash
# https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/who_what_benchmark

workspace=$HOME

cd openvino.genai/tools/who_what_benchmark/
pip install -e .

cd ${workspace}
pip install -r openvino.genai/tools/llm_bench/requirements.txt

pip uninstall openvino openvino-tokenizers openvino-genai -y
# pip install result/install_pkg/tools/openvino*
