cmake_minimum_required(VERSION 3.0)

# compile flags
set(CMAKE_CXX_FLAGS "-ta=tesla:managed -Minfo=accel -acc" CACHE STRING "CXX flags for all builds")
set(CMAKE_CXX_FLAGS_RELEASE "-fast" CACHE STRING "Release flags")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")

project(openacc_tests)

# write compile_commands.json file
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

set(SRC
    main.cpp
    )

add_executable(openacc_tests ${SRC})
set_property(TARGET openacc_tests PROPERTY CXX_STANDARD 17)
