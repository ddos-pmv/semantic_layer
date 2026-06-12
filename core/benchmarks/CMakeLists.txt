add_executable(onnx_bench main.cpp)

target_link_libraries(onnx_bench PRIVATE SemCore::SemCore)

if(TARGET ONNXRuntime::ONNXRuntime)
    target_link_libraries(onnx_bench PRIVATE ONNXRuntime::ONNXRuntime)
else()
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to an ONNX Runtime installation path, for example -DONNXRUNTIME_ROOT=/opt/homebrew/opt/onnxruntime")
endif()
