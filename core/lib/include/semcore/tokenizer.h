#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace semcore {

struct TokenizedText {
  std::vector<int64_t> input_ids;
  std::vector<int64_t> attention_mask;
  std::vector<int64_t> token_type_ids;
  int64_t seq_len = 0;

  friend std::ostream& operator<<(std::ostream& os, const TokenizedText& tokenized_text) {
    os << "input_ids[";
    for (auto id : tokenized_text.input_ids) {
      os << id << ", ";
    }

    os << "]\nattention_mask[";
    for (auto mask : tokenized_text.attention_mask) {
      os << mask << ", ";
    }

    os << "]\ntoken_type_ids[";
    for (auto id : tokenized_text.token_type_ids) {
      os << id << ", ";
    }
    os << "]\n";

    return os;
  }
};

class ITokenizer {
public:
  virtual ~ITokenizer() = default;

  virtual TokenizedText Tokenize(const std::string& text) const = 0;
};

class HfTokenizer : public ITokenizer {
public:
  explicit HfTokenizer(const std::string& path_to_json);
  ~HfTokenizer() override;

  TokenizedText Tokenize(const std::string& text) const override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace semcore
