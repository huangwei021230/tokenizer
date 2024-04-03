//#pragma once

#include "tokenization.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <map>

#define ll long long

namespace tokenizer{
    const std::unordered_map<std::wstring, std::wstring> VOCAB_FILES_NAMES = {
            {L"vocab_file",  L"vocab.json"},
            {L"merges_file", L"merges.txt"},
    };
    const std::unordered_map<std::wstring, size_t> PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
            {L"facebook/opt-125m", 1024},
    };

//    std::map<int, char> bytes_to_unicode() {
//        std::vector<int> bs;
//        std::vector<int> cs;
//        /**
//         *      Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
//                characters the bpe code barfs on.
//
//                The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
//                if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
//                decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
//                tables between utf-8 bytes and unicode strings.
//         */
//        for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i)
//            bs.push_back(i);
//
//        for (int i = static_cast<int>('¡'); i <= static_cast<int>('¬'); ++i)
//            bs.push_back(i);
//
//        for (int i = static_cast<int>('®'); i <= static_cast<int>('ÿ'); ++i)
//            bs.push_back(i);
//
//        cs = bs;
//
//        int n = 0;
//        for (int b = 0; b < 256; ++b) {
//            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
//                bs.push_back(b);
//                cs.push_back(256 + n);
//                ++n;
//            }
//        }
//
//        std::map<int, char> result;
//        for (size_t i = 0; i < bs.size(); ++i) {
//            result[bs[i]] = static_cast<char>(cs[i]);
//        }
//
//        return result;
//    }

    std::wstring strip(const std::wstring &text) {
        std::wstring ret = text;
        size_t pos = 0;
        while(pos < ret.size() && stripChar.find(ret[pos]) != std::string::npos) {
            pos++;
        }
        if(pos > 0) {
            ret = ret.substr(pos);
        }
        pos = ret.size() - 1;
        while(pos != (size_t) -1 && stripChar.find(ret[pos]) != std::string::npos) {
            pos--;
        }
        if(pos < ret.size() - 1) {
            ret = ret.substr(0, pos + 1);
        }
        return ret;
    }

    // deal with json Vocab
    std::shared_ptr<Vocab> loadVocab(const std::string &vocabFile) {
        std::shared_ptr<Vocab> vocab(new Vocab);
        size_t index = 0;
        std::ifstream ifs(vocabFile, std::ifstream::in);
        std::string line;
        while (getline(ifs, line)) {
            std::wstring token = convertToUnicode(line);
            if (token.empty())
                break;
            token = strip(token);
            (*vocab)[token] = index;
            index++;
        }
        return vocab;
    }



    GPT2Tokenizer::GPT2Tokenizer(
            const std::wstring &vocab_file,
            const std::wstring &merges_file,
            const size_t &vocab_size,
            const size_t &n_special,
            const std::wstring &unk_token,
            const std::wstring &bos_token,
            const std::wstring &eos_token,
            const std::wstring &pad_token,
            const bool &add_prefix_space,
            const bool &add_bos_token)
     {
        // initialize encoder and decoder
        std::ifstream file(vocab_file.c_str());
        if(!file.is_open()) {
            std::cerr << "Error opening file " << std::endl;
            exit(1);
        }
        nlohmann::json j;
        try {
            file >> j;
        } catch (nlohmann::json::parse_error &e) {
            std::cerr << "Error parsing json file " << std::endl;
            exit(1);
        }
        file.close();
        encoder = j;
        for(const auto & [key, value]: encoder) {
            std::cout << key << " : " << value << std::endl;
        }
    }

    std::vector<std::wstring> GPT2Tokenizer::tokenize(const std::wstring &text) const {
        return {};
    }


}



