// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/chat.h"

#include <string>
#include <vector>

namespace mini_llama {

void ChatSession::clear() {
    messages_.clear();
    token_history_.clear();
    prefix_cache_.clear();
    total_prompt_tokens_ = 0;
    total_generated_tokens_ = 0;
    total_time_ms_ = 0.0;
}

void ChatSession::addMessage(const std::string& role,
                             const std::string& content) {
    messages_.push_back({role, content});
}

void ChatSession::recordTurn(int prompt_tokens, int generated_tokens,
                             double time_ms) {
    total_prompt_tokens_ += prompt_tokens;
    total_generated_tokens_ += generated_tokens;
    total_time_ms_ += time_ms;
}

void ChatSession::setTokenHistory(const std::vector<int>& tokens) {
    token_history_ = std::move(tokens);
}

void ChatSession::appendToken(int token) { token_history_.push_back(token); }

void ChatSession::recordPrefix(const std::vector<int>& tokens) {
    prefix_cache_.insert(tokens);
}

size_t ChatSession::longestCachedPrefix(const std::vector<int>& tokens) const {
    return prefix_cache_.longestPrefix(tokens);
}

}  // namespace mini_llama
