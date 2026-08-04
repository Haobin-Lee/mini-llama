// Copyright (c) 2026
// SPDX-License-Identifier: MIT

// clang-format off
#include "mini_llama/tokenizer.h"
// clang-format on

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "errlog/bizlog.h"

namespace mini_llama {

using errlog::ErrorCode;

// Simple JSON array parser for the vocab format.
// Only supports: [{"id": n, "content": "...", "special": true/false}, ...]

namespace {

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) {
            return "";
        }
        size_t end = s.find_last_not_of(" \t\n\r");
        return s.substr(start, end - start + 1);
    }

    static std::string parseJsonString(const std::string& content, size_t& pos,
                                       bool& ok) {
        while (pos < content.size() && content[pos] != '"') {
            ++pos;
        }
        if (pos >= content.size() || content[pos] != '"') {
            BIZLOG(ErrorCode::kVocabParseError,
                   "expected opening quote for string");
            ok = false;
            return "";
        }
        ++pos;  // skip opening quote
        std::string result;
        while (pos < content.size() && content[pos] != '"') {
            if (content[pos] == '\\' && pos + 1 < content.size()) {
                char next = content[pos + 1];
                if (next == 'n') {
                    result += '\n';
                } else if (next == 't') {
                    result += '\t';
                } else if (next == 'r') {
                    result += '\r';
                } else if (next == '\\') {
                    result += '\\';
                } else if (next == '"') {
                    result += '"';
                } else if (next == 'u' && pos + 5 < content.size()) {
                    // Simple \uXXXX escape (4 hex digits)
                    std::string hex = content.substr(pos + 2, 4);
                    unsigned int codepoint = std::stoul(hex, nullptr, 16);
                    if (codepoint <= 0x7F) {
                        result += static_cast<char>(codepoint);
                    } else {
                        // For codepoints outside ASCII, just append as-is or
                        // use '?'
                        result += '?';
                    }
                    pos += 6;
                    continue;
                } else {
                    result += next;
                }
                pos += 2;
            } else {
                result += content[pos];
                ++pos;
            }
        }
        if (pos >= content.size()) {
            BIZLOG(ErrorCode::kVocabParseError, "unterminated string");
            ok = false;
            return result;
        }
        ++pos;  // skip closing quote
        return result;
    }

    static int parseJsonInt(const std::string& content, size_t& pos, bool& ok) {
        while (pos < content.size() &&
               !std::isdigit(static_cast<unsigned char>(content[pos])) &&
               content[pos] != '-') {
            ++pos;
        }
        if (pos >= content.size()) {
            BIZLOG(ErrorCode::kVocabParseError, "expected integer value");
            ok = false;
            return 0;
        }
        size_t start = pos;
        if (content[pos] == '-') {
            ++pos;
        }
        while (pos < content.size() &&
               std::isdigit(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        return std::stoi(content.substr(start, pos - start));
    }

    static bool parseJsonBool(const std::string& content, size_t& pos,
                              bool& ok) {
        if (content.substr(pos, 4) == "true") {
            pos += 4;
            return true;
        }
        if (content.substr(pos, 5) == "false") {
            pos += 5;
            return false;
        }
        BIZLOG(ErrorCode::kVocabParseError, "expected true or false");
        ok = false;
        return false;
    }

}  // namespace

JsonVocabTokenizer::JsonVocabTokenizer(const std::string& vocab_path) {
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
        BIZLOG(ErrorCode::kVocabParseError,
               "failed to open vocab: " + vocab_path);
        return;
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    if (f.bad()) {
        BIZLOG(ErrorCode::kVocabParseError,
               "failed to read vocab: " + vocab_path);
        return;
    }

    std::string content = buffer.str();

    // Find opening bracket
    size_t pos = content.find('[');
    if (pos == std::string::npos) {
        BIZLOG(ErrorCode::kVocabParseError, "vocab JSON must be an array");
        return;
    }
    ++pos;

    int max_id = -1;

    while (pos < content.size()) {
        // Find next object start
        while (pos < content.size() && content[pos] != '{') {
            if (content[pos] == ']') {
                break;
            }
            ++pos;
        }
        if (pos >= content.size() || content[pos] != '{') {
            break;
        }
        ++pos;

        VocabEntry entry;
        bool has_id = false;
        bool has_content = false;

        // Parse object fields
        while (pos < content.size() && content[pos] != '}') {
            // Skip whitespace and commas
            while (pos < content.size() &&
                   (std::isspace(static_cast<unsigned char>(content[pos])) ||
                    content[pos] == ',')) {
                ++pos;
            }
            if (pos >= content.size() || content[pos] == '}') {
                break;
            }

            // Parse key
            bool ok = true;
            std::string key = parseJsonString(content, pos, ok);
            if (!ok) {
                return;
            }

            // Skip to colon
            while (pos < content.size() && content[pos] != ':') {
                ++pos;
            }
            if (pos < content.size()) {
                ++pos;
            }
            // Skip whitespace
            while (pos < content.size() &&
                   std::isspace(static_cast<unsigned char>(content[pos]))) {
                ++pos;
            }

            if (key == "id") {
                entry.id = parseJsonInt(content, pos, ok);
                has_id = true;
            } else if (key == "content") {
                entry.content = parseJsonString(content, pos, ok);
                has_content = true;
            } else if (key == "special") {
                entry.special = parseJsonBool(content, pos, ok);
            } else {
                // Skip unknown value
                if (pos < content.size() && content[pos] == '"') {
                    parseJsonString(content, pos, ok);
                } else {
                    while (pos < content.size() && content[pos] != ',' &&
                           content[pos] != '}') {
                        ++pos;
                    }
                }
            }
            if (!ok) {
                return;
            }
        }

        if (pos < content.size() && content[pos] == '}') {
            ++pos;
        }

        if (!has_id || !has_content) {
            BIZLOG(ErrorCode::kVocabParseError,
                   "vocab entry missing id or content");
            return;
        }

        if (entry.id > max_id) {
            max_id = entry.id;
        }
        if (entry.id < 0) {
            BIZLOG(ErrorCode::kVocabParseError,
                   "vocab entry id must be non-negative");
            return;
        }

        if (entry.content == "<bos>") {
            bos_id_ = entry.id;
        } else if (entry.content == "<eos>") {
            eos_id_ = entry.id;
        } else if (entry.content == "<unk>") {
            unk_id_ = entry.id;
        }

        if (static_cast<size_t>(entry.id) >= id_to_entry_.size()) {
            id_to_entry_.resize(entry.id + 1);
        }
        id_to_entry_[entry.id] = entry;
        content_to_id_.push_back({entry.content, entry.id});
    }

    if (max_id < 0) {
        BIZLOG(ErrorCode::kVocabParseError, "vocab file contains no entries");
        return;
    }

    vocab_size_ = max_id + 1;

    // Sort content_to_id_ by content length descending for greedy longest match
    std::sort(content_to_id_.begin(), content_to_id_.end(),
              [](const auto& a, const auto& b) {
                  return a.first.size() > b.first.size();
              });

    valid_ = true;
}

std::vector<int> JsonVocabTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(bos_id_);

    size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        for (const auto& pair : content_to_id_) {
            const std::string& content = pair.first;
            if (content.empty()) {
                continue;
            }
            if (text.compare(pos, content.size(), content) == 0) {
                tokens.push_back(pair.second);
                pos += content.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            // No vocab entry matches this substring; consume one byte as UNK
            tokens.push_back(unk_id_);
            ++pos;
        }
    }

    return tokens;
}

std::string JsonVocabTokenizer::decodeToken(int token) const {
    if (token < 0 || static_cast<size_t>(token) >= id_to_entry_.size()) {
        return "<unk>";
    }
    return id_to_entry_[token].content;
}

std::string JsonVocabTokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        result += decodeToken(token);
    }
    return result;
}

std::unique_ptr<ITokenizer> createTokenizer(const std::string& vocab_path) {
    if (!vocab_path.empty()) {
        std::ifstream f(vocab_path);
        if (f.good()) {
            auto tok = std::make_unique<JsonVocabTokenizer>(vocab_path);
            if (tok->valid()) {
                return tok;
            }
            std::cerr << "Warning: failed to load vocab '" << vocab_path
                      << "', falling back to ASCII tokenizer" << std::endl;
        }
    }
    return std::make_unique<AsciiTokenizer>();
}

}  // namespace mini_llama
