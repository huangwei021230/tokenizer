//
// Created by PC on 2024/4/3.
//
#include "tokenization.h"

int main(){
    tokenizer::GPT2Tokenizer tokenizer(L"vocab.json", L"merges.txt", 1024, 0, L"<unk>");

    std::wstring text = L"Hello, my dog is cute";
//    std::vector<std::wstring> tokens = tokenizer.tokenize(text);
//    for (const auto &token : tokens) {
//        std::wcout << token << std::endl;
//    }

    return 0;
}