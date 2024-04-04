//#pragma once

#include "tokenization.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <algorithm>
#include <map>
#include <locale>
#include <codecvt>
#define ll long long

namespace tokenizer{
    const std::unordered_map<std::wstring, std::wstring> VOCAB_FILES_NAMES = {
            {L"vocab_file",  L"vocab.json"},
            {L"merges_file", L"merges.txt"},
    };
    const std::unordered_map<std::wstring, size_t> PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
            {L"facebook/opt-125m", 1024},
    };

    std::map<size_t , char> bytes_to_unicode() {
        std::vector<int> bs;
        std::vector<int> cs;
        for (unsigned char i = '!'; i <= '~'; ++i) {
            bs.push_back(i);
        }
        for (wchar_t i = L'¡'; i <= L'¬'; ++i) {
            bs.push_back(i);
        }
        for (wchar_t i = L'®'; i <= L'ÿ'; ++i) {
            bs.push_back(i);
        }
        cs = bs;
        int n = 0;
        for (int b = 0; b < (1 << 8); ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back((1 << 8) + n);
                n += 1;
            }
        }

        std::map<size_t , char> byte_unicode_map;
//        for (size_t i = 0; i < bs.size(); ++i) {
//            byte_unicode_map[bs[i]] = static_cast<char>(cs[i]);
//            std::cout<< bs[i] << " " << cs[i] << std::endl;
//        }
        return byte_unicode_map;
    }


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


//     deal with json Vocab
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
            const bool &add_bos_token){

        // initialize encoder and decoder
        std::filesystem::path absolute_path = std::filesystem::absolute(vocab_file);
        std::ifstream file(absolute_path);

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
        for(const auto &[key, value] : encoder) {
            decoder[value] = key;
        }

        // initialize byte encoder and decoder
        byte_encoder = bytes_to_unicode();
        for(const auto &[key, value] : byte_encoder) {
            byte_decoder[value] = key;
        }

        // deal with merge.txt
        std::filesystem::path absolute_path_merges = std::filesystem::absolute(merges_file);
        std::ifstream file_merges(absolute_path_merges);
        std::vector<std::string> bpe_merges;
        if(!file_merges.is_open()) {
            std::cerr << "Error opening file " << std::endl;
            exit(1);
        }
        std::string line;
        bool first_line = true;
        while (std::getline(file_merges, line)) {
            // skip the first line
            if (first_line) {
                first_line = false;
                continue;
            }
            // check if line is not empty before adding, to mimic Python's split("\n")[1:-1]
            if (!line.empty()) {
                bpe_merges.push_back(line);
            }
        }
        if (!bpe_merges.empty() && bpe_merges.back().empty()) {
            bpe_merges.pop_back();
        }





    }

    std::wstring GPT2Tokenizer::bpe(const std::wstring &token){
        if(cache.find(token) != cache.end()) {
            return cache[token];
        }
        std::wstring word = token;
        auto pairs = get_pairs(word);

    }
    std::vector<std::wstring> GPT2Tokenizer::tokenize(const std::wstring &text) const {
        std::vector<std::wstring> bpe_tokens;

        return {};
    }

    std::unordered_set<std::pair<char, char>> get_pairs(const std::string& word) {
        std::unordered_set<std::pair<char, char>> pairs;
        char prev_char = word[0];
        for (size_t i = 1; i < word.size(); ++i) {
            char current_char = word[i];
            pairs.insert(std::make_pair(prev_char, current_char));
            prev_char = current_char;
        }
        return pairs;
    }


}



