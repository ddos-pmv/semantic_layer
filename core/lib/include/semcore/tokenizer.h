#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace semcore {

struct TokenizedText {
	std::vector<int64_t> input_ids;
	std::vector<int64_t> attention_mask;
	std::vector<int64_t> token_type_ids;
	int64_t seq_len = 0;
	int64_t batch_size = 1;

	friend std::ostream& operator<<(std::ostream& os, const TokenizedText& tokenized_text) {
		os << "input_ids[";
		for (size_t i = 0; i < tokenized_text.input_ids.size(); ++i) {
			if (i == 0) os << " (";
			os << tokenized_text.input_ids[i];
			if ((i % tokenized_text.seq_len) == tokenized_text.seq_len - 1) os << "),";
		}

		os << "]\nattention_mask[";
		for (size_t i = 0; i < tokenized_text.attention_mask.size(); ++i) {
			if (i == 0) os << " (";
			os << tokenized_text.attention_mask[i];
			if ((i % tokenized_text.seq_len) == tokenized_text.seq_len - 1) os << "),";
		}

		os << "]\ntoken_type_ids[";
		for (size_t i = 0; i < tokenized_text.token_type_ids.size(); ++i) {
			if (i == 0) os << " (";
			os << tokenized_text.token_type_ids[i];
			if ((i % tokenized_text.seq_len) == tokenized_text.seq_len - 1) os << "),";
		}
		os << "]\n";

		return os;
	}
};

class ITokenizer {
   public:
	virtual ~ITokenizer() = default;

	virtual TokenizedText Tokenize(const std::string& text) const = 0;
	virtual TokenizedText TokenizeBatch(const std::vector<std::string>& text) const = 0;
};

class HfTokenizer : public ITokenizer {
   public:
	explicit HfTokenizer(std::string_view path_to_json);
	~HfTokenizer() override;

	TokenizedText Tokenize(const std::string& text) const override;
	TokenizedText TokenizeBatch(const std::vector<std::string>& texts) const override;

   private:
	class Impl;
	std::unique_ptr<Impl> impl_;
};

}  // namespace semcore
