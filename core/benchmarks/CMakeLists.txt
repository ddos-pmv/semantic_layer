


include(FetchContent)

FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark
        GIT_TAG origin/main
)

set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(benchmark)


FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)


add_executable(onnx_bench main.cpp)

target_link_libraries(onnx_bench PRIVATE
        SemCore::SemCore
        benchmark::benchmark
        nlohmann_json::nlohmann_json
)
