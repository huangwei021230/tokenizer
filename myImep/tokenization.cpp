#pragma once

#include "tokenization.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <map>
#define ll long long

namespace Tokenizer{
    const std::unordered_map<std::wstring, std::wstring> VOCAB_FILES_NAMES = {
            {L"vocab_file",  L"vocab.json"},
            {L"merges_file", L"merges.txt"},
    };
    const std::unordered_map<std::wstring, size_t> PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
            {L"facebook/opt-125m", 1024},
    };

    std::map<int, char> bytes_to_unicode() {
        std::vector<int> bs;
        std::vector<int> cs;
        /**
         *      Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
                characters the bpe code barfs on.

                The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
                if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
                decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
                tables between utf-8 bytes and unicode strings.
         */
        for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i)
            bs.push_back(i);

        for (int i = static_cast<int>('¡'); i <= static_cast<int>('¬'); ++i)
            bs.push_back(i);

        for (int i = static_cast<int>('®'); i <= static_cast<int>('ÿ'); ++i)
            bs.push_back(i);

        cs = bs;

        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }

        std::map<int, char> result;
        for (size_t i = 0; i < bs.size(); ++i) {
            result[bs[i]] = static_cast<char>(cs[i]);
        }

        return result;
    }

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


GPT2Tokenizer::GPT2Tokenizer(const std::wstring &vocab_file, const std::wstring &merges_file,
                                        const size_t &vocab_size, const size_t &n_special,
                                        const std::wstring &unk_token, const std::wstring &bos_token,
                                        const std::wstring &eos_token, const std::wstring &pad_token,
                                        const bool &add_prefix_space, const bool &add_bos_token) {
    // initialize encoder and decoder
    std::map<std::string, std::string> encoder;
    std::ifstream vocab_handle(vocab_file);
    if (vocab_handle.is_open()) {
        std::string line;
        while (std::getline(vocab_handle, line)) {
            size_t delimiter_pos = line.find('\t');
            if (delimiter_pos != std::string::npos) {
                std::string token = line.substr(0, delimiter_pos);
                std::string id = line.substr(delimiter_pos + 1);
                encoder[id] = token;
            }
        }
        vocab_handle.close();
    }
}

}


