//#pragma once

#include "tokenization.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <regex>
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
#include <iomanip>
// #include <boost/regex.hpp>
#define ll long long

namespace tokenizer{
    const std::unordered_map<std::wstring, std::wstring> VOCAB_FILES_NAMES = {
            {L"vocab_file",  L"vocab.json"},
            {L"merges_file", L"merges.txt"},
    };
    const std::unordered_map<std::wstring, size_t> PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
            {L"facebook/opt-125m", 1024},
    };

    std::vector<std::wstring> split(const std::wstring &text) {
        std::vector<std::wstring> result;
        size_t pos = 0, found;
        while ((found = text.find_first_not_of(stripChar, pos)) != std::wstring::npos) {
            pos = text.find_first_of(stripChar, found);
            if (pos == std::wstring::npos) {
                result.push_back(text.substr(found));
                break;
            }
            result.push_back(text.substr(found, pos - found));
        }
        return result;
    }

    std::vector<std::string> split(const std::string &text) {
        std::vector<std::string> result;
        size_t pos = 0, found;
        std::istringstream ir(text);
        std::string buffer;
        while(std::getline(ir, buffer, ' ')) {
            result.push_back(buffer);
        }
        return result;
    }

    std::map<size_t , wchar_t> bytes_to_unicode() {
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

        std::map<size_t , wchar_t> byte_unicode_map;
       for (size_t i = 0; i < bs.size(); ++i) {
           byte_unicode_map[bs[i]] = static_cast<wchar_t>(cs[i]);
       }

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


    //   deal with json Vocab
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
            const std::string &vocab_file,
            const std::string &merges_file,
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
        encoder = std::make_shared<Vocab>();
        decoder = std::make_shared<InvVocab>();
        for(const auto &[key, value] : j.items()) {
            std::wstring token = convertToUnicode(key);
            (*encoder)[token] = value;
            (*decoder)[value] = token;
        }

        // initialize byte encoder and decoder
        byte_encoder = bytes_to_unicode();
        for(const auto &[key, value] : byte_encoder) {
            byte_decoder[value] = key;
        }

        // deal with merge.txt
        std::filesystem::path absolute_path_merges = std::filesystem::absolute(merges_file);
        std::ifstream file_merges(absolute_path_merges, std::ios::in | std::ios::binary);
        // open file encoding = utf-8
        file_merges.imbue(std::locale(file_merges.getloc(), new std::codecvt_utf8<wchar_t>));


        // TODO: transform this into Unicode form
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
                // remove the first space
                bpe_merges.push_back(line);
            }
        }
        if (!bpe_merges.empty() && bpe_merges.back().empty()) {
            bpe_merges.pop_back();
        }
        file_merges.close();


        // tuple
        std::vector<std::vector<std::wstring>> bpe_merge_words;
        for(const auto &line : bpe_merges) {
            std::wstring wline = convertToUnicode(line);
            std::vector<std::wstring> words = split(wline);
            bpe_merge_words.push_back(words);
        }

        for(int i = 0;i< bpe_merge_words.size(); ++i) {
            bpe_ranks[std::make_pair(bpe_merge_words[i][0], bpe_merge_words[i][1])] = i;
        }

        // regex pattern
        pat = std::regex("'s|'t|'re|'ve|'m|'ll|'d| ?\\w+| ?\\d+| ?[^\\s\\w\\d]+|\\s+(?!\\S)|\\s+");
        
    }
    size_t GPT2Tokenizer::getVocabId(const std::wstring &token) {
        auto it = (*encoder).find(token);
        if(it != (*encoder).end()) {
            return it->second;
        } else {
            auto unknown_it = (*encoder).find(L"<unk>");
            if(it != (*encoder).end()) {
                return unknown_it->second;
            }else{
                std::cerr << "Error: token not found in encoder" << std::endl;
                exit(1);
            }
        }
    }
    // std::wstring GPT2Tokenizer::bpe(const std::wstring &token){
    //     if(cache.find(token) != cache.end()) {
    //         return cache[token];
    //     }
    //     std::vector<std::wstring> word;
        
    //     for(auto &c : token) {
    //         word.push_back(std::wstring(1, c));
    //     }
    //     auto pairs = get_pairs(word);
    //     if(pairs.empty()) {
    //         return token;
    //     }
    //     while (true) {
    //         auto it = std::min_element(pairs.begin(), pairs.end(), [this](const auto& pair1, const auto& pair2) {
    //             return bpe_ranks[pair1] < bpe_ranks[pair2];
    //         });
    //         auto bigram = *it;
    //         if (bpe_ranks.find(make_pair(bigram.first, bigram.second)) == bpe_ranks.end()) {
    //             break;
    //         }
    //         std::string new_word;
    //         size_t i = 0;
    //         while (i < word.size()) {
    //             auto j = word.find(bigram.first, i);
    //             if (j == std::string::npos) {
    //                 new_word.append(word.substr(i));
    //                 break;
    //             }
    //             new_word.append(word.substr(i, j - i));
    //             i = j;
    //             if (word[i] == bigram.first[0] && i < word.size() - 1 && word[i + 1] == bigram.second[0]) {
    //                 new_word.append(bigram.first + bigram.second);
    //                 i += 2;
    //             } else {
    //                 new_word.push_back(word[i]);
    //                 ++i;
    //             }
    //         }
    //         word = new_word;
    //         if (word.size() == 1) {
    //             break;
    //         } else {
    //             pairs = get_pairs(word);
    //         }
    //     }
    //     cache[token] = word;
    //     return word;
    // }

    
    std::vector<std::wstring> GPT2Tokenizer::tokenize(const std::string &text) const {;
        std::string text_copy = text;
        std::vector<std::wstring> bpe_tokens;
        // Tokenize the text
        // Match results
        std::smatch match;
        // Search for matches in the text
        std::string::const_iterator start = text_copy.begin();
        std::string::const_iterator end = text_copy.end();
        while (std::regex_search(start, end, match, pat)) {
            // Update the start iterator to search for next match
            std::string utf8_token = match.str();
            start = match[0].second;
            std::wstringstream result;
            for (char c : utf8_token) {  
                if (byte_encoder.find(c) != byte_encoder.end()) {
                    auto it = byte_encoder.find(static_cast<size_t>(c));
                    result << it->second;
                }
                else
                {
                    result << static_cast<wchar_t>(c);
                }
            }
            bpe_tokens.push_back(result.str());
            //TODO: bpe
        }
        return bpe_tokens;
    }
    std::vector<size_t> GPT2Tokenizer::convertTokensToIds(const std::vector<std::wstring> &tokens) {
        std::vector<size_t> token_ids;
        for(const auto &token : tokens) {
            auto it = (*encoder).find(token);
            if(it != (*encoder).end()) {
                token_ids.push_back(it->second);
            } else {
                auto unknown_it = (*encoder).find(L"<unk>");
                if(it != (*encoder).end()) {
                    token_ids.push_back(unknown_it->second);
                }else{
                    std::cerr << "Error: token not found in encoder" << std::endl;
                    exit(1);
                }
            }
        }
        return token_ids;
    }
    std::unordered_set<std::pair<std::wstring, std::wstring>, pair_hash> GPT2Tokenizer::get_pairs(const std::vector<std::wstring>& words){
        std::unordered_set<std::pair<std::wstring, std::wstring>, pair_hash> pairs;
        for(size_t i = 1; i < words.size(); ++i) {
            pairs.insert(std::make_pair(words[i - 1], words[i]));
        }
        return pairs;
    }

    std::vector<std::string> GPT2Tokenizer::convertIdsToTokens(const std::vector<size_t> &ids) {
        std::vector<std::string> tokens;
        for(const auto &id : ids) {
            auto it = (*decoder).find(id);
            if(it != (*decoder).end()) {
                std::stringstream result;
                std::wstring unicode_token = it->second;
                for (wchar_t c : unicode_token) {
                    if (byte_decoder.find(c) != byte_decoder.end()) {
                        auto iter = byte_decoder.find(static_cast<wchar_t>(c));
                        result << static_cast<char>(iter->second);
                    }
                    else
                    {
                        result << static_cast<char>(c);
                    }
                }
                tokens.push_back(result.str());

            } else {
                auto unknown_it = (*decoder).find(0);
                if(it != (*decoder).end()) {
                    tokens.push_back(convertFromUnicode(unknown_it->second));
                }else{
                    std::cerr << "Error: token not found in decoder" << std::endl;
                    exit(1);
                }
            }
        }
        return tokens;
    }


}



