#include "opt_preprocessor.h"
#include <iostream>

namespace OPT {
OptPreprocessor::OptPreprocessor(const std::string &vocab_file, const std::string &merges_file) {
    tokenizer_ = std::make_unique<tokenizer::GPT2Tokenizer>(vocab_file, merges_file, 1024, 0, L"<unk>");
}

torch::Tensor OptPreprocessor::preprocessText(const std::string& text) {
    std::vector<size_t> all_tokens;

    // tokenize the text
    all_tokens.emplace_back(tokenizer_->getVocabId(L"</s>"));
    auto ids = tokenizer_->convertTokensToIds(tokenizer_->tokenize(text));
    if (ids.size() > text_length - 2) {
        ids.resize(text_length - 2);
    }
    all_tokens.insert(all_tokens.end(), ids.begin(), ids.end());

    // convert all_tokens to tensor
    auto tensor_text = torch::zeros({1, text_length}, torch::kInt32);
    for (auto i = 0; i < all_tokens.size(); i++) {
        tensor_text[0][i] = all_tokens[i];
    }

    // std::cout << "tensor size: " << tensor_text.sizes() << std::endl;

    return tensor_text;
}

}