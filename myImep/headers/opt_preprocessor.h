// #pragma once
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <tokenization.h>

namespace OPT {
class OptPreprocessor {
public:
    OptPreprocessor(const std::string &vocab_file, const std::string &merges_file);
    torch::Tensor preprocessText(const std::string &text);

    constexpr static int text_length = 52;

private:
   std::unique_ptr<tokenizer::GPT2Tokenizer> tokenizer_;
};
} // namespace OPT
