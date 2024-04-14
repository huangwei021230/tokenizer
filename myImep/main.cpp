#include <iostream>
#include <fstream>
#include <chrono>
#include "opt_infer_engine.h"
#include "tokenization.h"

int main() {
    std::cout << "initializing..." << std::endl;
    auto gpt2Tokenizer = tokenizer::GPT2Tokenizer("../vocab.json", "../merges.txt",50272, 0, L"<unk>");
    std::string text = "huangwei, huangwei askdjlasjdlkas lkjasld";
    auto ids = gpt2Tokenizer.convertTokensToIds(gpt2Tokenizer.tokenize(text));

    std::cout << "ids size: " << ids.size() << std::endl;
    for (auto &id : ids) {
        std::cout << id << " ";
    }
//    std::ofstream outfile("execution_times.txt");
//
//    for (int i = 0; i < 50; ++i) {
//        auto start = std::chrono::high_resolution_clock::now();
//        std::string return_string = opt_infer_engine.getReturnString(text);
//        auto end = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//        std::cout << return_string << std::endl;
//        outfile << "Execution " << i + 1 << ": " << duration.count() << " microseconds" << std::endl;
//    }
//
//    outfile.close();

    return 0;
}
