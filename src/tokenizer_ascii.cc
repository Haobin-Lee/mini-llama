// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include <stdexcept>

#include "mini_llama/tokenizer.h"

namespace mini_llama {

std::vector<int> AsciiTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(bosId());
    // 逐字节编码，将ascii码<128的字符，直接作为token_id，>=128的字符映射到unk_id
    for (unsigned char c : text) {
        int id = static_cast<int>(c);
        if (id >= kVocabSize) {
            tokens.push_back(unkId());
        } else {
            tokens.push_back(id);
        }
    }
    return tokens;
}
std::string AsciiTokenizer::decodeToken(int token) const {
    if (token == bosId()) {
        return "<bos>";
    }
    if (token == eosId()) {
        return "<eos>";
    }
    if (token == unkId()) {
        return "<unk>";
    }
    if (token >= kVocabSize) {
        return "<unk>";
    }
    if (token < 32 || token == 127) {
        // 控制字符，直接转为空字符串
        return "";
    }
    return std::string(1, static_cast<char>(token));
}
std::string AsciiTokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        result += decodeToken(token);
    }
    return result;
}

}  // namespace mini_llama
