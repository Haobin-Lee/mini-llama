// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_TERMINAL_H_
#define INCLUDE_MINI_LLAMA_TERMINAL_H_

#include <string>

#include "mini_llama/chat.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

// Interactive terminal I/O for chat mode.
class Terminal {
public:
    Terminal() = default;

    void printUserPrompt() const;
    std::string readLine() const;  // empty on EOF
    void printAssistantPrefix() const;
    void printTokenText(const std::string& text) const;
    void flush() const;
    void newLine() const;
    void printHelp() const;
    void printStats(const ChatSession& session) const;
    void printParams(const SamplingParams& params) const;
    void printMessage(const std::string& msg) const;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_TERMINAL_H_
