#include <semcore/tokenizer.h>

#include <tokenizers_cpp.h>

#include <fstream>
#include <stdexcept>
#include <utility>

namespace semcore {
namespace details {

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    throw std::runtime_error("Cannot open tokenizer file: " + path);
  }

  std::string data;
  fs.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);

  return data;
}

} // namespace details

class HfTokenizer::Impl {
public:
  explicit Impl(const std::string& tokenizer_json_path) {
    auto blob = details::LoadBytesFromFile(tokenizer_json_path);
    tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);

    if (!tok_) {
      throw std::runtime_error("Failed to load tokenizer from " + tokenizer_json_path);
    }
  }

  TokenizedText Encode(const std::string& prompt) const {
    auto encoded = tok_->Encode(prompt);
    const auto len = encoded.size();

    TokenizedText result;
    result.input_ids = std::move(encoded);
    result.attention_mask.assign(len, 1);
    result.token_type_ids.assign(len, 0);
    result.seq_len = static_cast<int32_t>(len);

    return result;
  }

private:
  std::unique_ptr<tokenizers::Tokenizer> tok_;
};

HfTokenizer::HfTokenizer(const std::string& path_to_json)
    : impl_(std::make_unique<Impl>(path_to_json)) {}

HfTokenizer::~HfTokenizer() = default;

TokenizedText HfTokenizer::Tokenize(const std::string& text) const {
  return impl_->Encode(text);
}

} // namespace semcore
