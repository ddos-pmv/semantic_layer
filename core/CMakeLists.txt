cmake_minimum_required(VERSION 3.20)
project(SemCore VERSION 0.1.0 LANGUAGES CXX)

option(SEMCORE_BUILD_EXAMPLES "Build SemCore examples" OFF)
option(SEMCORE_BUILD_TESTS "Build SemCore tests" ON)
option(SEMCORE_BUILD_BENCHS "Build SemCore benchmarks" ON)
option(SEMCORE_FETCH_DEPENDENCIES "Fetch SemCore third-party dependencies" OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

set(TOKENIZERS_CPP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp")
add_subdirectory("${TOKENIZERS_CPP_PATH}" tokenizers EXCLUDE_FROM_ALL)
add_subdirectory(lib)

if(SEMCORE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SEMCORE_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(SEMCORE_BUILD_BENCHS)
    add_subdirectory(benchmarks)
endif()
