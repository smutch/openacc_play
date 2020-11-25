.PHONY: build
build:
	cmake --build build -- ${MAKE_ARGS}
	./build/cusolver_tests

.PHONY: reconfig
reconfig:
	rm -rf build
	# CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/targets/x86_64-linux CXX=`which pgc++` cmake -S. -Bbuild ${CMAKE_ARGS} -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda/include
	CXX=`which nvc++` cmake -S. -Bbuild ${CMAKE_ARGS}
