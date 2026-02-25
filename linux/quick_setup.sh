# conda create -n sparse_attn_test python==3.12.* -y
# conda activate sparse_attn_test
# pip install -r requirements.txt
# pip install openvino====2026.1.0.0.dev202600206 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
# pip install openvino-tokenizers==2026.1.0.0.dev20260206 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
# pip install openvino-genai==2026.1.0.0.dev20260206 --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

# https://10-211-120-125.iotg.sclab.intel.com/openvino_ci/private_builds/dldt/master/pre_commit
# python -m venv sparse_attn_test
# source sparse_attn_test/bin/activate

cd $HOME/openvino.genai/tools/who_what_benchmark
pip install -e .
pip uninstall openvino -y
pip uninstall openvino-genai -y
pip uninstall openvino-tokenizers -y

wget -r -l 0 -nH --cut-dirs=1  --no-parent --reject="index.html*" --no-check-certificate https://10-211-120-125.iotg.sclab.intel.com/openvino_ci/private_builds/dldt/master/pre_commit/a1b941784abf61320fad77be7a981e46f6b34665/private_linux_ubuntu_24_04_release/wheels/openvino-2026.1.0.dev20260210-20691-cp312-cp312-manylinux_2_39_x86_64.whl
wget -r -l 0 -nH --cut-dirs=1  --no-parent --reject="index.html*" --no-check-certificate https://10-211-120-125.iotg.sclab.intel.com/openvino_ci/private_builds/dldt/master/pre_commit/a1b941784abf61320fad77be7a981e46f6b34665/private_linux_ubuntu_24_04_release/wheels/openvino_genai-2026.1.0.0.dev20260210-2097-cp312-cp312-linux_x86_64.whl
wget -r -l 0 -nH --cut-dirs=1  --no-parent --reject="index.html*" --no-check-certificate https://10-211-120-125.iotg.sclab.intel.com/openvino_ci/private_builds/dldt/master/pre_commit/a1b941784abf61320fad77be7a981e46f6b34665/private_linux_ubuntu_24_04_release/wheels/openvino_tokenizers-2026.1.0.0.dev20260210-534-py3-none-linux_x86_64.whl

pip install private_builds/dldt/master/pre_commit/a1b941784abf61320fad77be7a981e46f6b34665/private_linux_ubuntu_24_04_release/wheels/openvino*.whl
