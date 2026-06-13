#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string_view>

namespace semcore {

class BaseModel {
public:
  BaseModel(
      std::string_view onnx_path,
      Ort::Env& env,
      const Ort::SessionOptions& session_options)
      : session_(env, onnx_path.data(), session_options) {}

  virtual ~BaseModel() = default;

  virtual void DumpModelInfo(std::ostream& os = std::cout) {
    Ort::AllocatorWithDefaultOptions allocator;

    DumpTensorInfo("output", session_.GetOutputCount(), allocator, os);
    DumpTensorInfo("input", session_.GetInputCount(), allocator, os);
  }

private:
  void DumpTensorInfo(
      std::string_view label,
      size_t count,
      Ort::AllocatorWithDefaultOptions& allocator,
      std::ostream& os) {
    for (size_t i = 0; i < count; ++i) {
      auto info = label == "output"
          ? session_.GetOutputTypeInfo(i)
          : session_.GetInputTypeInfo(i);
      auto tensor_info = info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      auto name = label == "output"
          ? session_.GetOutputNameAllocated(i, allocator)
          : session_.GetInputNameAllocated(i, allocator);

      os << label << "[" << i << "] " << name.get() << "\n";
      os << "  type: " << tensor_info.GetElementType() << "\n";
      os << "  shape: ";

      for (auto dim : shape) {
        os << dim << " ";
      }

      os << "\n";
    }
  }

  Ort::Session session_;
};

} // namespace semcore
