// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_
#define INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_

#include <string>
#include <vector>

#include "mini_llama/chat.h"

namespace mini_llama {

// Turn a chat message history into a single prompt string.
// Plain mode (default): "System:/User:/Assistant:" layout.
// Template mode: applies a model-specific template (e.g. Qwen2).
// A trailing assistant prefix cues the model to start generating.
class PromptBuilder {
public:
    PromptBuilder() = default;

    void setChatTemplate(const std::string& template_str);
    std::string build(const std::vector<ChatMessage>& messages) const;

private:
    std::string chat_template_;

    std::string buildPlain(const std::vector<ChatMessage>& messages) const;
    std::string buildQwen2(const std::vector<ChatMessage>& messages) const;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_
