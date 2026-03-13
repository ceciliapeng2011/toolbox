dump_base=$HOME/openvino/dump
mkdir -p "$dump_base"

OV_GPU_PAGED_ATTENTION_IMPL=ocl cl_cache_dir=$HOME/.cache/cl_cache cliloader -cdt --dump-dir "$dump_base/ocl_basic_0" $HOME/openvino//intel64/Release/ov_gpu_unit_tests --gtest_filter=smoke_paged_attention_perf_ocl/paged_attention_test.basic/0 --gtest_repeat=3 --gtest_color=no
OV_GPU_PAGED_ATTENTION_IMPL=ocl cl_cache_dir=$HOME/.cache/cl_cache cliloader -cdt --dump-dir "$dump_base/ocl_basic_1" $HOME/openvino//intel64/Release/ov_gpu_unit_tests --gtest_filter=smoke_paged_attention_perf_ocl/paged_attention_test.basic/1 --gtest_repeat=3 --gtest_color=no
OV_GPU_PAGED_ATTENTION_IMPL=ocl cl_cache_dir=$HOME/.cache/cl_cache cliloader -cdt --dump-dir "$dump_base/ocl_basic_2" $HOME/openvino//intel64/Release/ov_gpu_unit_tests --gtest_filter=smoke_paged_attention_perf_ocl/paged_attention_test.basic/2 --gtest_repeat=3 --gtest_color=no

OV_GPU_PAGED_ATTENTION_IMPL=cm cl_cache_dir=$HOME/.cache/cm_cache cliloader -cdt --dump-dir "$dump_base/cm_basic_0" $HOME/openvino//intel64/Release/ov_gpu_unit_tests --gtest_filter=smoke_paged_attention_perf_cm/xattention_test.basic/0 --gtest_repeat=3 --gtest_color=no
OV_GPU_PAGED_ATTENTION_IMPL=cm cl_cache_dir=$HOME/.cache/cm_cache cliloader -cdt --dump-dir "$dump_base/cm_basic_1" $HOME/openvino//intel64/Release/ov_gpu_unit_tests --gtest_filter=smoke_paged_attention_perf_cm/xattention_test.basic/1 --gtest_repeat=3 --gtest_color=no
OV_GPU_PAGED_ATTENTION_IMPL=cm cl_cache_dir=$HOME/.cache/cm_cache cliloader -cdt --dump-dir "$dump_base/cm_basic_2" $HOME/openvino//intel64/Release/ov_gpu_unit_tests --gtest_filter=smoke_paged_attention_perf_cm/xattention_test.basic/2 --gtest_repeat=3 --gtest_color=no