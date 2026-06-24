#include <onnxruntime/onnxruntime_cxx_api.h>
#include <semcore/model.h>
#include <semcore/tokenizer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace semcore {
namespace {

Ort::SessionOptions CreateSessionOptions(const EmbeddingModelConfig& config) {
	Ort::SessionOptions options;
	options.SetIntraOpNumThreads(config.intra_op_threads);
	options.SetInterOpNumThreads(config.inter_op_threads);
	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	// options.SetLogSeverityLevel(0);
	if (!config.enable_mem_pattern) {
		options.DisableMemPattern();
	}
	//
	// // Настройка CoreML Execution Provider непосредственно при создании опций
	// std::unordered_map<std::string, std::string> coreml_options;
	// coreml_options["ModelFormat"] = "MLProgram";
	// coreml_options["MLComputeUnits"] = "ALL";
	// coreml_options["RequireStaticInputShapes"] = "0";
	//
	// try {
	// 	options.AppendExecutionProvider("CoreML", coreml_options);
	// 	std::cout << "[semcore] CoreML Execution Provider успешно добавлен.\n";
	// } catch (const Ort::Exception& e) {
	// 	std::cerr << "[semcore] Предупреждение: CoreML недоступен, откат на CPU. Ошибка: " << e.what() << "\n";
	// }

	// Добавляем CPU как гарантированный fallback
	// options.AppendExecutionProvider_CPU( 0 );
	// options.AppendExecutionProvider_CPU(0);

	return options;
}

void Normalize(std::span<float> embedding) {
	const float squared_norm =
		std::inner_product(embedding.begin(), embedding.end(), embedding.begin(), 0.0F);
	if (squared_norm == 0.0F) {
		return;
	}

	const float inverse_norm = 1.0F / std::sqrt(squared_norm);
	for (auto& value : embedding) {
		value *= inverse_norm;
	}
}

void NormalizeBatch(EmbeddingBatch& batch) {
	for (size_t i = 0; i < batch.BatchSize(); ++i) {
		auto row = batch[i];

		Normalize(row);
	}
}

}  // namespace

class EmbeddingModel::Impl {
   public:
	explicit Impl(const EmbeddingModelConfig& config)
		: config_(config),
		  env_(ORT_LOGGING_LEVEL_WARNING, "semcore_embedding_model"),
		  session_options_(CreateSessionOptions(config)),
		  tokenizer_(config.tokenizer_path),
		  session_(env_, config_.onnx_path.c_str(), session_options_) {}

	std::vector<float> Encode(const std::string& text) {
		auto tokenized = tokenizer_.Tokenize(text);
		auto [embedding, embedding_dim] = Run(tokenized);

		if (config_.normalize_embeddings) {
			Normalize(embedding);
		}

		return embedding;
	}

	EmbeddingBatch EncodeBatch(const std::vector<std::string>& texts) {
		auto tokenized = tokenizer_.TokenizeBatch(texts);
		auto [data, embedding_dim] = Run(tokenized);

		EmbeddingBatch batch{
			.embeddings = std::move(data),
			.embedding_dim = embedding_dim,
		};

		if (config_.normalize_embeddings) {
			NormalizeBatch(batch);
		}

		return batch;
	}

   private:
	std::pair<std::vector<float>, size_t> Run(const TokenizedText& text) {
		if (text.seq_len <= 0) {
			throw std::runtime_error("Cannot encode empty token sequence");
		}

		if (text.batch_size <= 0) {
			throw std::runtime_error("Cannot encode empty batch");
		}

		std::array<const char*, 3> input_names{"input_ids", "attention_mask", "token_type_ids"};
		std::array<const char*, 1> output_names{"last_hidden_state"};

		std::array<int64_t, 2> input_shape{text.batch_size, text.seq_len};
		std::array<Ort::Value, 3> input_tensors{
			Ort::Value::CreateTensor<int64_t>(memory_info_, const_cast<int64_t*>(text.input_ids.data()),
											  text.input_ids.size(), input_shape.data(), input_shape.size()),
			Ort::Value::CreateTensor<int64_t>(memory_info_, const_cast<int64_t*>(text.attention_mask.data()),
											  text.attention_mask.size(), input_shape.data(),
											  input_shape.size()),
			Ort::Value::CreateTensor<int64_t>(memory_info_, const_cast<int64_t*>(text.token_type_ids.data()),
											  text.token_type_ids.size(), input_shape.data(),
											  input_shape.size())};

		auto outputs = session_.Run(run_options_, input_names.data(), input_tensors.data(),
									input_tensors.size(), output_names.data(), output_names.size());
		auto& last_hidden_state = outputs[0];
		auto tensor_info = last_hidden_state.GetTensorTypeAndShapeInfo();
		auto shape = tensor_info.GetShape();

		// Official ONNX builds output raw token embeddings [batch, seq, dim];
		// reduce them to sentence embeddings with the same masked mean pooling
		// the Python side uses. L2 normalization stays optional (config flag).
		if (shape.size() != 3 || shape[0] != text.batch_size || shape[1] != text.seq_len) {
			throw std::runtime_error("Unexpected last_hidden_state shape");
		}

		const auto batch_size = static_cast<size_t>(shape[0]);
		const auto seq_len = static_cast<size_t>(shape[1]);
		const auto embedding_dim = static_cast<size_t>(shape[2]);
		const float* data = last_hidden_state.GetTensorData<float>();

		std::vector<float> pooled(batch_size * embedding_dim, 0.0F);
		for (size_t b = 0; b < batch_size; ++b) {
			float* out = pooled.data() + b * embedding_dim;
			float mask_sum = 0.0F;
			for (size_t t = 0; t < seq_len; ++t) {
				const auto mask = static_cast<float>(text.attention_mask[b * seq_len + t]);
				if (mask == 0.0F) {
					continue;
				}
				mask_sum += mask;
				const float* token = data + (b * seq_len + t) * embedding_dim;
				for (size_t d = 0; d < embedding_dim; ++d) {
					out[d] += token[d] * mask;
				}
			}
			const float inverse = 1.0F / std::max(mask_sum, 1e-9F);
			for (size_t d = 0; d < embedding_dim; ++d) {
				out[d] *= inverse;
			}
		}

		return {std::move(pooled), embedding_dim};
	}

	EmbeddingModelConfig config_;
	Ort::Env env_;
	Ort::SessionOptions session_options_;
	HfTokenizer tokenizer_;
	Ort::Session session_;
	Ort::RunOptions run_options_;
	Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

EmbeddingModel::EmbeddingModel(const EmbeddingModelConfig& config) : impl_(std::make_unique<Impl>(config)) {}

EmbeddingModel::~EmbeddingModel() = default;

EmbeddingModel::EmbeddingModel(EmbeddingModel&&) noexcept = default;

EmbeddingModel& EmbeddingModel::operator=(EmbeddingModel&&) noexcept = default;

std::vector<float> EmbeddingModel::Encode(const std::string& text) const {
	return impl_->Encode(text);
}

EmbeddingBatch EmbeddingModel::EncodeBatch(const std::vector<std::string>& texts) const {
	return impl_->EncodeBatch(texts);
}

}  // namespace semcore
