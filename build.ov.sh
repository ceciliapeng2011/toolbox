# RelWithDebInfo   ./build.sh RelWithDebInfo.DEBUGOFF.ITTON OFF ON
#     -DCMAKE_CXX_FLAGS="-Werror" \
    # -DCMAKE_CXX_FLAGS="-Wno-error=nonnull" \
    # cmake --build build_Debug --target clang_format_fix_all -j8
(
git submodule update --init --recursive
mkdir -p build_$1 &&\
cd build_$1 &&\
echo $1 $2 $3 \
cmake -UCPACK_GENERATOR .. &&\
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_INTEL_MYRIAD_COMMON=OFF \
    -DCMAKE_CXX_FLAGS="-Wno-error=nonnull -Wno-error=undef -Werror=unused-variable" \
    -DENABLE_INTEL_GNA=OFF \
    -DENABLE_OPENCV=OFF \
    -DENABLE_CPPLINT=ON \
    -DENABLE_CLANG_FORMAT=ON \
    -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
    -DENABLE_SYSTEM_SNAPPY=ON \
    -DBUILD_nvidia_plugin=OFF \
    -DENABLE_NCC_STYLE=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_FUNCTIONAL_TESTS=ON -DENABLE_DATA=OFF \
    -DENABLE_CPU_SPECIFIC_TARGET_PER_TEST=ON \
    -DENABLE_INTEL_CPU=ON \
    -DENABLE_INTEL_GPU=ON \
    -DENABLE_INTEL_NPU=OFF \
    -DENABLE_DRIVER_COMPILER_ADAPTER=OFF \
    -DENABLE_TEMPLATE=OFF \
    -DENABLE_AUTO=OFF \
    -DENABLE_HETERO=OFF \
    -DENABLE_MULTI=OFF \
    -DENABLE_AUTO_BATCH=OFF \
    -DENABLE_PROFILING_ITT=$3 \
    -DENABLE_SAMPLES=ON \
    -DENABLE_PYTHON=ON \
    -DENABLE_JS=OFF \
    -DENABLE_OV_ONNX_FRONTEND=OFF \
    -DENABLE_OV_PADDLE_FRONTEND=OFF \
    -DENABLE_OV_TF_FRONTEND=ON \
    -DENABLE_OPENVINO_DEBUG=OFF \
    -DENABLE_CPU_DEBUG_CAPS=OFF \
    -DENABLE_DEBUG_CAPS=$2 \
    -DENABLE_GPU_DEBUG_CAPS=$2 \
    -DENABLE_CPU_PROFILER=OFF \
    -DENABLE_ONEDNN_FOR_GPU=ON \
    -DCMAKE_INSTALL_PREFIX=`pwd`/install && \
cmake --build . --parallel $(nproc) && \
cmake --install . \
)


