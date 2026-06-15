#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace semcore {

struct EmbeddingModelConfig {
	std::string onnx_path;
	std::string tokenizer_path;
	int intra_op_threads = 1;
	int inter_op_threads = 1;
	bool enable_mem_pattern = true;
	bool normalize_embeddings = false;
};

struct EmbeddingBatch {
	std::vector<float> embeddings;
	size_t embedding_dim;

	std::span<const float> operator[](size_t index) const {
		return {embeddings.data() + index * embedding_dim, embedding_dim};
	}

	std::span<float> operator[](size_t index) {
		return {embeddings.data() + index * embedding_dim, embedding_dim};
	}

	size_t BatchSize() const {
		return embeddings.size() / embedding_dim;
	}
};

class EmbeddingModel {
   public:
	explicit EmbeddingModel(const EmbeddingModelConfig& config);
	~EmbeddingModel();

	EmbeddingModel(EmbeddingModel&&) noexcept;
	EmbeddingModel& operator=(EmbeddingModel&&) noexcept;

	EmbeddingModel(const EmbeddingModel&) = delete;
	EmbeddingModel& operator=(const EmbeddingModel&) = delete;

	std::vector<float> Encode(const std::string& text) const;
	EmbeddingBatch EncodeBatch(const std::vector<std::string>& texts) const;

   private:
	class Impl;
	std::unique_ptr<Impl> impl_;
};

}  // namespace semcore
