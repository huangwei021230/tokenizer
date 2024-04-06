//
// Created by PC on 2024/4/3.
//
#include "tokenization.h"

int main(){
    tokenizer::GPT2Tokenizer tokenizer("../vocab.json", "../merges.txt", 1024, 0, L"<unk>");
    std::string text = "Hello, my dog's 2 feet tall.";
    auto tokens = tokenizer.tokenize(text);
//    std::vector<std::wstring> tokens = tokenizer.tokenize(text);
//    for (const auto &token : tokens) {
//        std::wcout << token << std::endl;
//    }

    return 0;
}