#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string_view>

namespace semcore {

class BaseModel {
public:
  BaseModel(std::string_view onnx_path, Ort::Env &env,
            const Ort::SessionOptions &session_options)
      : session_(env, onnx_path.data(), session_options) {}

  virtual ~BaseModel() = default;

  virtual void DumpModelInfo(std::ostream &os = std::cout) {
    Ort::AllocatorWithDefaultOptions allocator;

    DumpTensorInfo("output", session_.GetOutputCount(), allocator, os);
    DumpTensorInfo("input", session_.GetInputCount(), allocator, os);
  }

protected:
  void DumpTensorInfo(std::string_view label, size_t count,
                      Ort::AllocatorWithDefaultOptions &allocator,
                      std::ostream &os) {
    for (size_t i = 0; i < count; ++i) {
      auto info = label == "output" ? session_.GetOutputTypeInfo(i)
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
  Ort::RunOptions run_options_;
  Ort::MemoryInfo memory_info_ =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

class EmbeddingModel : public BaseModel {
public:
  EmbeddingModel(std::string_view onnx_path, Ort::Env &env,
                 const Ort::SessionOptions &session_options)
      : BaseModel(onnx_path, env, session_options) {}

  std::vector<float> Encode(const TokenizedText &text) {

    std::array<const char *, 3> input_names{"input_ids", "attention_mask",
                                            "token_type_ids"};

    std::array<const char *, 2> output_names{"token_embeddings",
                                             "sentence_embedding"};

    const int64_t batch_size = 1;
    std::array<int64_t, 2> input_shape{batch_size, text.seq_len};
    std::array<Ort::Value, 3> input_tensors{
        Ort::Value::CreateTensor<int64_t>(
            memory_info_, const_cast<int64_t *>(text.input_ids.data()),
            text.input_ids.size(), input_shape.data(), input_shape.size()),

        Ort::Value::CreateTensor<int64_t>(
            memory_info_, const_cast<int64_t *>(text.attention_mask.data()),
            text.attention_mask.size(), input_shape.data(), input_shape.size()),
        Ort::Value::CreateTensor<int64_t>(
            memory_info_, const_cast<int64_t *>(text.attention_mask.data()),
            text.attention_mask.size(), input_shape.data(),
            input_shape.size())};

    auto outputs = session_.Run(run_options_, input_names.data(),
                                input_tensors.data(), input_tensors.size(),
                                output_names.data(), output_names.size());

    auto &sentence_embedding = outputs[1];

    auto tensor_info = sentence_embedding.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    if (shape.size() != 2 || shape[0] != 1) {
      throw std::runtime_error("Unexpected sentence_embedding shape");
    }

    const size_t embedding_dim = static_cast<size_t>(shape[1]);

    const float *data = sentence_embedding.GetTensorData<float>();

    return std::vector<float>(data, data + embedding_dim);
  }
};

} // namespace semcore
