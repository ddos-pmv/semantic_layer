cmake_minimum_required(VERSION 3.20)
project(SemCore VERSION 0.1.0 LANGUAGES CXX)

option(SEMCORE_BUILD_EXAMPLES "Build SemCore examples" OFF)
option(SEMCORE_BUILD_TESTS "Build SemCore tests" ON)
option(SEMCORE_BUILD_BENCHS "Build SemCore benchs" ON)
option(SEMCORE_FETCH_DEPENDENCIES "Fetch SemCore third-party dependencies" OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

# --- Fetch DEPS ---
if(SEMCORE_FETCH_DEPENDENCIES)

include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)


FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)


endif()


# --- add LIB(target) ---
add_library(SemCore INTERFACE)
add_library(SemCore::SemCore ALIAS SemCore)

target_compile_features(SemCore INTERFACE cxx_std_23)

target_include_directories(
    SemCore
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)


# --- link ORT ---
if(ONNXRUNTIME_ROOT)
    find_library(
        ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib"
        NO_DEFAULT_PATH
        REQUIRED
    )

    add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
    set_target_properties(
        ONNXRuntime::ONNXRuntime
        PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
endif()


# --- build EXMAPLES ---

if(SEMCORE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()



# --- build TESTS ---
if(SEMCORE_BUILD_TESTS)
    add_subdirectory(tests)
endif()




# --- build BENCHS ---
if(SEMCORE_BUILD_BENCHS)
    add_subdirectory(benchmarks)
endif()


