//
// Created by PC on 2024/4/3.
//
#include "tokenization.h"
#include "opt_infer_engine.h"
#
int main(){
    OPT::OptInferEngine opt_infer_engine("../pytorch_model.bin", "../vocab.json", "../merges.txt");
    std::string text = "Hello, my dog's 2 feet tall.";
    std::string return_string = opt_infer_engine.getReturnString(text);
    std::cout << return_string << std::endl;
//    std::vector<std::wstring> tokens = tokenizer.tokenize(text);
//    for (const auto &token : tokens) {
//        std::wcout << token << std::endl;
//    }

    return 0;
}