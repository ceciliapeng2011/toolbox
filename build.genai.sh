git submodule update --init --recursive
source $HOME/openvino/build_Debug/install/setupvars.sh
cmake -DCMAKE_BUILD_TYPE=Debug -S ./ -B ./build/
cmake --build ./build/ --config Debug -j 8
cmake --install ./build/ --config Debug --prefix $HOME/openvino/build_Debug/install
