#include <semcore/tokenizer.h>
#include <tokenizers_cpp.h>

#include <format>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace semcore {
namespace details {

std::string LoadBytesFromFile(std::string_view path) {
	std::ifstream fs(path.data(), std::ios::in | std::ios::binary);
	if (fs.fail()) {
		throw std::runtime_error(std::format("Cannot open tokenizer file: {}", path));
	}

	std::string data;
	fs.seekg(0, std::ios::end);
	const auto size = static_cast<size_t>(fs.tellg());
	fs.seekg(0, std::ios::beg);
	data.resize(size);
	fs.read(data.data(), size);

	return data;
}

}  // namespace details

class HfTokenizer::Impl {
   public:
	explicit Impl(std::string_view tokenizer_json_path) {
		auto blob = details::LoadBytesFromFile(tokenizer_json_path);
		tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);

		if (!tok_) {
			throw std::runtime_error(std::format("Failed to load tokenizer from {}", tokenizer_json_path));
		}
	}

	TokenizedText Encode(const std::string& prompt) const {
		auto encoded = tok_->Encode(prompt);
		const auto len = encoded.size();

		TokenizedText result;
		result.input_ids.assign(encoded.begin(), encoded.end());
		result.attention_mask.assign(len, 1);
		result.token_type_ids.assign(len, 0);
		result.seq_len = static_cast<int64_t>(len);

		return result;
	}
	TokenizedText EncodeBatch(const std::vector<std::string>& prompts) const {
		auto encoded = tok_->EncodeBatch(prompts);
		const auto len = encoded.size();
		size_t max_len = 0;
		for (const auto& elem : encoded) {
			max_len = std::max(max_len, elem.size());
		}

		TokenizedText result;
		result.input_ids.assign(max_len * len, 0);
		result.attention_mask.assign(max_len * len, 0);
		result.token_type_ids.assign(max_len * len, 0);
		result.seq_len = static_cast<int64_t>(max_len);
		result.batch_size = len;

		for (size_t i = 0; i < len; i++) {
			const auto offset = max_len * i;
			const auto cur_len = encoded[i].size();
			std::copy(encoded[i].begin(), encoded[i].end(), result.input_ids.begin() + offset);
			std::fill_n(result.attention_mask.begin() + offset, cur_len, 1);
		}

		return result;
	}

   private:
	std::unique_ptr<tokenizers::Tokenizer> tok_;
};

HfTokenizer::HfTokenizer(std::string_view path_to_json) : impl_(std::make_unique<Impl>(path_to_json)) {}

HfTokenizer::~HfTokenizer() = default;

TokenizedText HfTokenizer::Tokenize(const std::string& text) const {
	return impl_->Encode(text);
}
TokenizedText HfTokenizer::TokenizeBatch(const std::vector<std::string>& texts) const {
	return impl_->EncodeBatch(texts);
}

}  // namespace semcore
