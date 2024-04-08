// #pragma once
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <tokenization.h>

namespace OPT {
class OptProcessor {
public:
    OptProcessor(const std::string &vocab_file, const std::string &merges_file);
    torch::Tensor preprocessText(const std::string &text);
    std::vector<std::string> getReturnString(torch::Tensor tensor);

private:
   std::unique_ptr<tokenizer::GPT2Tokenizer> tokenizer_;
};
} // namespace OPT
