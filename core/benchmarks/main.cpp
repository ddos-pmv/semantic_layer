#include "semcore/tokenizer.h"

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <semcore/model.h>

#include <iostream>
#include <string>
#include <string_view>

int main() {
  constexpr std::string_view model_path =
      "../models/paraphrase-multilingual-MiniLM-L12-v2/model.onnx";

  const std::string tokenizer_json_path =
      "../models/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json";

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "semantic_onnx_test");

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  semcore::EmbeddingModel model(model_path, env, session_options);

  model.DumpModelInfo();

  semcore::HfTokenizer tokenizer(tokenizer_json_path);

  auto tokens =
      tokenizer.Tokenize("Some something someone somehow sometimes hate you");
  std::cout << tokens << std::endl;

  auto embedding = model.Encode(tokens);
  std::cout << "embedding dim: " << embedding.size() << "\n";
  std::cout << "first value: " << embedding[0] << "\n";

  return 0;
}
