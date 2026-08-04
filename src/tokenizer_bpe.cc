// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <fstream>
#include <sstream>

#include "errlog/bizlog.h"
#include "mini_llama/tokenizer.h"
#include "nlohmann/json.hpp"  // 添加 nlohmann_json 头文件

namespace mini_llama {

using namespace errlog;
// ---------------------------------------------------------------------------
// UTF-8 codepoint mapping
// ---------------------------------------------------------------------------

static std::string codepointToUtf8(char32_t cp) {
    std::string result;
    if (cp <= 0x7F) {
        result += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}

// Extract next UTF-8 codepoint as a string; return empty if invalid
static std::string nextUtf8Codepoint(const std::string& s, size_t& pos) {
    if (pos >= s.size()) {
        return "";
    }
    unsigned char first = s[pos];
    size_t len = 1;
    if ((first & 0x80) == 0) {
        len = 1;
    } else if ((first & 0xE0) == 0xC0) {
        len = 2;
    } else if ((first & 0xF0) == 0xE0) {
        len = 3;
    } else if ((first & 0xF8) == 0xF0) {
        len = 4;
    } else {
        // Invalid UTF-8, skip one byte
        ++pos;
        return "";
    }
    if (pos + len > s.size()) {
        ++pos;
        return "";
    }
    std::string result = s.substr(pos, len);
    pos += len;
    return result;
}

// ---------------------------------------------------------------------------
// GPT-2 bytes_to_unicode mapping
// ---------------------------------------------------------------------------

static std::vector<std::string> buildBytesToUnicode() {
    std::vector<std::string> map(256);

    std::vector<int> bs;
    for (int c = '!'; c <= '~'; ++c) {
        bs.push_back(c);
    }
    for (int c = 0xA1; c <= 0xAC; ++c) {
        bs.push_back(c);
    }
    for (int c = 0xAE; c <= 0xFF; ++c) {
        bs.push_back(c);
    }

    std::vector<int> cs = bs;
    int n = 0;
    // 缺失的字节映射到256+ 区域
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        map[bs[i]] = codepointToUtf8(static_cast<char32_t>(cs[i]));
    }
    return map;
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------

void BpeTokenizer::buildByteMappings() {
    b2u_ = buildBytesToUnicode();
    u2b_.clear();
    for (int b = 0; b < 256; ++b) {
        u2b_[b2u_[b]] = static_cast<uint8_t>(b);
    }
}

bool BpeTokenizer::Load(const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_path) {
    buildByteMappings();

    if (!loadVocab(vocab_path) || !loadMerges(merges_path) ||
        !loadSpecialTokens(special_path)) {
        return false;
    }

    BIZLOG(ErrorCode::kLoadBPE, vocab_.size(), merge_ranks_.size(), bos_id_,
           eos_id_, unk_id_);
    return true;
}

// Load vocab.json: { "token": id, ... }
bool BpeTokenizer::loadVocab(const std::string& vocab_path) {
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
        BIZLOG(ErrorCode::kOpenFileFailed, vocab_path);
        return false;
    }

    nlohmann::json json_file = nlohmann::json::parse(f);

    if (json_file.is_discarded() || !json_file.is_object()) {
        BIZLOG(ErrorCode::kLoadJsonFailed, vocab_path);
        return false;
    }

    for (const auto& item : json_file.items()) {
        const std::string& token = item.key();
        const nlohmann::json& val = item.value();

        // 校验value是否整数
        if (!val.is_number_integer()) {
            BIZLOG(ErrorCode::kInvalidVocabID, token);
            continue;
        }
        int token_id = val.get<int>();
        vocab_[token] = token_id;
    }

    if (vocab_.empty()) {
        BIZLOG(ErrorCode::kEmptyVocab);
        return false;
    }

    int max_id = 0;
    for (const auto& kv : vocab_) {
        if (kv.second > max_id) {
            max_id = kv.second;
        }
    }

    // 构建词汇ID到文本的映射
    id_to_token_.resize(max_id + 1);
    for (const auto& kv : vocab_) {
        id_to_token_[kv.second] = kv.first;
    }

    // Extract special tokens (e.g. <|im_start|>) from vocab for
    // longest-match Encode.
    for (const auto& kv : vocab_) {
        if (kv.first.size() >= 4 && kv.first.front() == '<' &&
            kv.first[1] == '|' && kv.first[kv.first.size() - 2] == '|' &&
            kv.first.back() == '>') {
            special_tokens_.push_back({kv.first, kv.second});
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto& a, const auto& b) {
                  return a.first.size() > b.first.size();
              });

    return true;
}

// Load merges.txt: "token1 token2" per line
bool BpeTokenizer::loadMerges(const std::string& merges_path) {
    std::ifstream f(merges_path);
    if (!f.is_open()) {
        BIZLOG(ErrorCode::kOpenFileFailed, merges_path);
        return false;
    }
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string a, b;
        if (ss >> a >> b) {
            merge_ranks_[{a, b}] = rank++;
        }
    }
    return true;
}

// Load special_tokens.json
bool BpeTokenizer::loadSpecialTokens(const std::string& special_path) {
    std::ifstream f(special_path);
    if (f.is_open()) {
        auto json_file = nlohmann::json::parse(f);
        if (json_file.is_discarded() || !json_file.is_object()) {
            BIZLOG(ErrorCode::kLoadJsonFailed, special_path);
            return false;
        }

        auto find_id = [&json_file](const std::string& token,
                                    const nlohmann::json& item) {
            int id;
            if (!item.contains(token) && item[token].is_number_integer()) {
                return item[token].get<int>();
            }

            return -1;
        };

        for (const auto& item : json_file.items()) {
            if (!item.value().is_number_integer()) {
                BIZLOG(ErrorCode::kInvalidVocabID, item.key());
                continue;
            }
            if (item.key() == "bos_id") {
                bos_id_ = item.value().get<int>();
            } else if (item.key() == "eos_id") {
                eos_id_ = item.value().get<int>();
            } else if (item.key() == "unk_id") {
                unk_id_ = item.value().get<int>();
            }
        }
    }

    return true;
}

std::vector<int> BpeTokenizer::encode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }

    std::vector<int> result;
    size_t pos = 0;

    while (pos < text.size()) {
        // Try to match a special token first (longest match)
        bool matched = false;
        for (const auto& st : special_tokens_) {
            const std::string& token_str = st.first;
            if (pos + token_str.size() <= text.size() &&
                std::memcmp(text.data() + pos, token_str.data(),
                            token_str.size()) == 0) {
                result.push_back(st.second);
                pos += token_str.size();
                matched = true;
                break;
            }
        }
        if (matched) {
            continue;
        }

        // Find the next special token or end of string
        size_t end = text.size();
        for (const auto& st : special_tokens_) {
            size_t p = text.find(st.first, pos);
            if (p != std::string::npos && p < end) {
                end = p;
            }
        }

        // Encode the plain text segment [pos, end) with BPE
        std::string segment = text.substr(pos, end - pos);
        pos = end;

        // 1. 文本转换为词汇（字节到Unicode映射）
        std::vector<std::string> word;
        for (unsigned char c : segment) {
            if (c >= b2u_.size()) {
                BIZLOG(ErrorCode::kByteOutOfRange, c, b2u_.size());
                return {};
            }
            word.push_back(b2u_[c]);
        }

        // 2. 应用BPE合并
        while (word.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_idx = word.size();  // invalid

            // rank最小的优先级最高
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                auto it = merge_ranks_.find({word[i], word[i + 1]});
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }

            // 在merge.txt中找不到可以合并的词汇对，中止合并
            if (best_idx >= word.size()) {
                break;  // no more merges possible
            }

            // 在word中找到了优先级最高的词汇对，合并词汇对
            word[best_idx] += word[best_idx + 1];
            word.erase(word.begin() + best_idx + 1);
        }

        // 3. 词汇列表转换为ID
        for (const auto& w : word) {
            auto it = vocab_.find(w);
            if (it != vocab_.end()) {
                result.push_back(it->second);
            } else {
                result.push_back(unk_id_);
            }
        }
    }

    return result;
}

std::string BpeTokenizer::decodeToken(int token) const {
    if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
        return id_to_token_[token];
    }
    return "";
}

std::string BpeTokenizer::decode(const std::vector<int>& tokens) const {
    // 1. 词汇ID转换为文本
    std::string text;
    for (int id : tokens) {
        if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
            text += id_to_token_[id];
        }
    }

    // 2. Unicode字符转换为字节
    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t start = pos;
        std::string cp = nextUtf8Codepoint(text, pos);
        if (cp.empty()) {
            // Invalid UTF-8, skip
            if (pos == start) {
                ++pos;
            }
            continue;
        }
        auto it = u2b_.find(cp);
        if (it != u2b_.end()) {
            result += static_cast<char>(it->second);
        } else {
            // Not a byte token; pass through as-is (shouldn't happen for
            // standard BPE)
            result += cp;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<ITokenizer> createBpeTokenizer(
    const std::string& vocab_path, const std::string& merges_path,
    const std::string& special_path) {
    auto tok = std::make_unique<BpeTokenizer>();
    if (tok->Load(vocab_path, merges_path, special_path)) {
        return tok;
    }
    return nullptr;
}

}  // namespace mini_llama