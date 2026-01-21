@REM C:\ceciliapeng\openvino.venv\Scripts\activate

set OpenVINO_DIR=C:\ceciliapeng\openvino\release_install\runtime
set TBB_DIR=C:\ceciliapeng\openvino\temp\Windows_AMD64\tbb\bin

@REM set no_proxy=localhost,127.0.0.0/8,::1
@REM set ftp_proxy=http://child-prc.intel.com:913/

set https_proxy=http://proxy-dmz.intel.com:912
set http_proxy=http://proxy-dmz.intel.com:911

git clone https://github.com/openvinotoolkit/openvino.genai.git

cd openvino.genai

git submodule update --init --recursive

cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build/
cmake --build ./build/ --config Release -j10

cmake --install ./build/ --config Release --prefix C:\ceciliapeng\openvino\release_install