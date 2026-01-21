@REM set no_proxy=localhost,127.0.0.0/8,::1
@REM set ftp_proxy=http://child-prc.intel.com:913/

@REM set https_proxy=http://proxy-dmz.intel.com:912
@REM set http_proxy=http://proxy-dmz.intel.com:911


@REM C:\ceciliapeng\openvino.venv\Scripts\activate

@REM git clone https://github.com/openvinotoolkit/openvino.git

cd openvino

git submodule update --init --recursive

@REM cmake -S . -B build -G "Visual Studio 18 2026" -A x64


@REM cmake -S . -B build ^
@REM     -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF  ^
@REM     -DCMAKE_CXX_FLAGS="%CMAKE_CXX_FLAGS% /we4267"

cmake -S . -B build -G "Visual Studio 18 2026" -A x64  ^
    -Wno-dev -DENABLE_CPPLINT=OFF -DENABLE_CLANG_FORMAT=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF  ^
    -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF  ^
    -DCMAKE_BUILD_TYPE=Release -DENABLE_PROFILING_ITT=OFF  ^
    -DENABLE_INTEL_GPU=ON -DENABLE_INTEL_CPU=ON -DENABLE_INTEL_GNA=OFF -DENABLE_MULTI=OFF -DENABLE_AUTO=ON -DENABLE_AUTO_BATCH=OFF -DENABLE_HETERO=OFF  -DENABLE_INTEL_NPU=OFF  ^
    -DENABLE_TEMPLATE=OFF -DENABLE_TEMPLATE_REGISTRATION=OFF  ^
    -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=ON  ^
    -DENABLE_PYTHON=ON -DENABLE_WHEEL=OFF  ^
    -DENABLE_TESTS=OFF -DENABLE_FUNCTIONAL_TESTS=OFF  ^
    -DTHREADING=TBB -DENABLE_SYSTEM_TBB=ON -DENABLE_JS=OFF  ^
    -DENABLE_OV_ONNX_FRONTEND=ON  ^
    -DENABLE_GPU_DEBUG_CAPS=ON -DENABLE_DEBUG_CAPS=ON  ^
    -DCMAKE_INSTALL_PREFIX=C:\ceciliapeng\openvino\release_install

cmake --build ./build --config Release --parallel 16
cmake --install ./build --prefix C:\ceciliapeng\openvino\release_install --config Release

cd ..