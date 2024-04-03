//
// Created by PC on 2024/4/3.
//

#ifndef TOKENIZER_TOKENIZATION_H
#define TOKENIZER_TOKENIZATION_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
// #include <boost/algorithm/string.hpp>

const std::wstring stripChar = L" \t\n\r\f\v";
using Vocab = std::unordered_map<std::wstring, size_t>;
using InvVocab = std::unordered_map<size_t , std::wstring>;

namespace tokenizer {
static const std::unordered_map<std::wstring, std::wstring> VOCAB_FILES_NAMES;
static const std::unordered_map<std::wstring, size_t> PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES;


    GPT2Tokenizer(
            const std::wstring &vocab_file,
            const std::wstring &merges_file,
            const size_t &vocab_size,
            const size_t &n_special,
            const std::wstring &unk_token,
            const std::wstring &bos_token= L"<|endoftext|>",
            const std::wstring &eos_token= L"<|endoftext|>",
            const std::wstring &pad_token= L"",
            const bool &add_prefix_space=false,
            const bool &add_bos_token=false);

};


#endif //TOKENIZER_TOKENIZATION_H
