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
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <memory>
// #include <boost/algorithm/string.hpp>
// #include <boost/regex.hpp>
#include <utf8proc.h>
#include <map>

// hash for pair<char,char>
struct pair_hash {
    inline std::size_t operator()(const std::pair<char,char> & v) const {
        return v.first*31+v.second;
    }
};
// strings
const std::wstring stripChar = L" \t\n\r\f\v";

// typedef
using Vocab = std::unordered_map<std::wstring, size_t>;
using InvVocab = std::unordered_map<size_t , std::wstring>;

namespace tokenizer {


static std::vector<std::wstring> split(const std::wstring &text);


class GPT2Tokenizer {
public:
    GPT2Tokenizer(
            const std::wstring &vocab_file,
            const std::wstring &merges_file,
            const size_t &vocab_size,
            const size_t &n_special,
            const std::wstring &unk_token,
            const std::wstring &bos_token = L"<|endoftext|>",
            const std::wstring &eos_token = L"<|endoftext|>",
            const std::wstring &pad_token = L"",
            const bool &add_prefix_space = false,
            const bool &add_bos_token = false);
    std::vector<std::wstring> tokenize(const std::string &text) const;
    std::wstring bpe(const std::wstring &token);
    std::unordered_set<std::pair<char, char>, pair_hash> get_pairs(const std::wstring &word);
private:
    std::shared_ptr<Vocab> mVocab;
    std::shared_ptr<InvVocab> mInvVocab;
    std::map<std::string, std::size_t> encoder;
    std::map<std::size_t , std::string> decoder;
    std::map<std::size_t, char> byte_encoder;
    std::map<char, std::size_t> byte_decoder;
    std::map<std::vector<std::wstring>, std::size_t> bpe_ranks;
    std::unordered_map<std::wstring, std::wstring> cache;
    std::regex pat;
};

static std::wstring convertToUnicode(const std::string &text) {
    size_t i = 0;
    std::wstring ret;
    while (i < text.size()) {
        wchar_t codepoint;
        utf8proc_ssize_t forward =
                utf8proc_iterate((utf8proc_uint8_t *)&text[i], text.size() - i, (utf8proc_int32_t *)&codepoint);
        if (forward < 0)
            return L"";
        ret += codepoint;
        i += forward;
    }
    return ret;
}

static std::string convertFromUnicode(const std::wstring &wText) {
    char dst[64];
    std::string ret;
    for (auto ch : wText) {
        utf8proc_ssize_t num = utf8proc_encode_char(ch, (utf8proc_uint8_t *)dst);
        if (num <= 0)
            return "";
        ret += std::string(dst, dst + num);
    }
    return ret;
}



};


#endif //TOKENIZER_TOKENIZATION_H
