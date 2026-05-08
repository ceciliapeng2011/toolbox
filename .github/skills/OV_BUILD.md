# openvino venv
source ~/openvino.venv/bin/activate

# openvino build
cd ~/openvino
find build_Debug/ -name cm |xargs rm -rf
 ./build.ov.sh Debug --gpu-only

 # gpu plugin unit test of xattention
 ./bin/intel64/Debug/ov_gpu_unit_tests --gtest_filter=*xatt*
## example
OV_VERBOSE=4 ./bin/intel64/Debug/ov_gpu_unit_tests --gtest_filter=smoke_cm_xattention/xattention_test.basic/89 2>&1 | tee debug_output.log
