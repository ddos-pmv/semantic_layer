#include <semcore/model.h>

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

int main() {
	semcore::EmbeddingModelConfig config{
		.onnx_path = "../models/paraphrase-multilingual-MiniLM-L12-v2/model.onnx",
		.tokenizer_path = "../models/paraphrase-multilingual-MiniLM-L12-v2/tokenizer.json",
	};

	semcore::EmbeddingModel model(config);

	auto embedding = model.Encode("Some something someone somehow sometimes hate you");
	std::cout << "embedding dim: " << embedding.size() << "\n";
	std::cout << "first value: " << embedding[0] << "\n";

	std::vector<std::string_view> texts{
		"Some something someone somehow sometimes hate you",
		"Claim your prize by clicking this link",
	};
	auto batch = model.EncodeBatch(texts);
	std::cout << "batch size: " << batch.size() << "\n";

	return 0;
}
