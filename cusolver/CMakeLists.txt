cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0074 NEW)

# compile flags
set(CMAKE_CXX_FLAGS "-ta=tesla:managed -Minfo=accel -acc" CACHE STRING "CXX flags for all builds")
set(CMAKE_CXX_FLAGS_RELEASE "-fast" CACHE STRING "Release flags")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")

project(cusolver_tests CXX)

# write compile_commands.json file
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

set(SRC
    main.cpp
    )

add_executable(cusolver_tests ${SRC})
set_property(TARGET cusolver_tests PROPERTY CXX_STANDARD 17)

find_package(fmt REQUIRED)
target_link_libraries(cusolver_tests PRIVATE fmt::fmt)

# Can't get this to work on Ubuntu with CUDA toolkit + NVIDA HPC SDK
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(cusolver_tests PRIVATE CUDA::cusolver)

target_link_directories(cusolver_tests PRIVATE /usr/local/cuda/lib64)
target_link_libraries(cusolver_tests PRIVATE cusolver cudart)
