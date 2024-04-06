// #pragma once
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <tokenization.h>

namespace OPT {
class OptPreprocessor {
public:
    OptPreprocessor(const std::string &vocab_file);
    torch::Tensor preprocessImage(const std::string &image_path);
    torch::Tensor preprocessText(const std::string &text);

    static std::vector<float> image_mean;
    static std::vector<float> image_std;
    constexpr static int image_size = 224;
    constexpr static int text_length = 52;

private:
    std::unique_ptr<Tokenizer::FullTokenizer> tokenizer_;
};
} // namespace OPT