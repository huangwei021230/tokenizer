#include "opt_processor.h"
#include <iostream>
#include "string"
namespace OPT {
OptProcessor::OptProcessor(const std::string &vocab_file, const std::string &merges_file) {
    tokenizer_ = std::make_unique<tokenizer::GPT2Tokenizer>(vocab_file, merges_file, 50272, 0, L"<unk>");
}

torch::Tensor OptProcessor::preprocessText(const std::string& text) {
    std::vector<size_t> all_tokens;
    // tokenize the text
    all_tokens.emplace_back(tokenizer_->getVocabId(L"</s>"));
    auto ids = tokenizer_->convertTokensToIds(tokenizer_->tokenize(text));

//    std::cout << "ids size: " << ids.size() << std::endl;
    all_tokens.insert(all_tokens.end(), ids.begin(), ids.end());

    // convert all_tokens to tensor
    auto tensor_text = torch::zeros({1, static_cast<long>(all_tokens.size())}, torch::kInt32);
    for (auto i = 0; i < all_tokens.size(); i++) {
        tensor_text[0][i] = all_tokens[i];
    }

//    std::cout << "tensor size: " << tensor_text.sizes() << std::endl;


    return tensor_text;
}

    std::vector<std::string> OptProcessor::getReturnString(torch::Tensor tensor) {
        auto data = tensor.accessor<long, 1>();
        // 创建一个 size_t 类型的 vector，并将数据复制到其中
        std::vector<size_t> vec;
        for (size_t i = 0; i < tensor.size(0); ++i) {
            // 将 LongType 数据转换为 size_t 类型，并确保在范围内
            size_t value = static_cast<size_t>(data[i]);
            if (value < 0 || value > 50272) {
                throw std::runtime_error("Data out of range [0, 50272]");
            }
            vec.push_back(value);
        }

        std::vector<std::string> tokens = tokenizer_->convertIdsToTokens(vec);
        return tokens;
    }

}