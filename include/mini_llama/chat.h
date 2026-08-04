// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CHAT_H_
#define INCLUDE_MINI_LLAMA_CHAT_H_

#include <string>
#include <vector>

#include "mini_llama/radix_tree.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

struct ChatMessage {
    std::string role;  // "system", "user", "assistant"
    std::string content;
};

// State of an interactive chat session.
class ChatSession {
public:
    ChatSession() = default;

    void clear();
    void addMessage(const std::string& role, const std::string& content);
    void recordTurn(int prompt_tokens, int generated_tokens, double time_ms);
    void setTokenHistory(const std::vector<int>& tokens);
    void recordPrefix(const std::vector<int>& tokens);
    void appendToken(int token);

    size_t longestCachedPrefix(const std::vector<int>& tokens) const;
    const std::vector<ChatMessage>& getMessages() const { return messages_; }
    const std::vector<int>& getTokenHistory() const { return token_history_; }
    const RadixTree& getPrefixCache() const { return prefix_cache_; }

    const SamplingParams& getSamplingParams() const { return sampling_params_; }
    int getTotalPromptTokens() const { return total_prompt_tokens_; }
    int getTotalGeneratedTokens() const { return total_generated_tokens_; }
    double getTotalTime() const { return total_time_ms_; }

    void setMessages(const std::vector<ChatMessage>& messages) {
        messages_ = std::move(messages);
    }
    void setSamplingParams(const SamplingParams& params) {
        sampling_params_ = std::move(params);
    }
    void setTotalPromptTokens(int tokens) { total_prompt_tokens_ = tokens; }
    void setTotalGeneratedTokens(int tokens) {
        total_generated_tokens_ = tokens;
    }
    void setTotalTime(double time) { total_time_ms_ = time; }

private:
    std::vector<ChatMessage> messages_;
    std::vector<int> token_history_;
    RadixTree prefix_cache_;
    SamplingParams sampling_params_;

    int total_prompt_tokens_ = 0;
    int total_generated_tokens_ = 0;
    double total_time_ms_ = 0.0;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CHAT_H_
