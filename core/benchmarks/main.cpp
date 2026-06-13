#include "semcore/tokenizer.h"

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <semcore/model.h>

#include <iostream>
#include <string>
#include <string_view>

int main() {
  constexpr std::string_view model_path =
      "../models/paraphrase-multilingual-MiniLM-L12-v2/model.onnx";

  const std::string tokenizer_json_path = "../models/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json";

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "semantic_onnx_test");

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);


  semcore::BaseModel model(model_path, env, session_options);

  model.DumpModelInfo();

  semcore::HfTokenizer tokenizer(tokenizer_json_path);

  auto result = tokenizer.Tokenize("Test sentence");
  std::cout << result << std::endl;

  return 0;
}
