// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// chat 测试：ChatSession 状态机 + PromptBuilder（plain/qwen2/自定义模板）。

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "mini_llama/chat.h"
#include "mini_llama/prompt_builder.h"
#include "tests/test_main.h"

using mini_llama::ChatMessage;
using mini_llama::ChatSession;
using mini_llama::PromptBuilder;
using mini_llama::SamplingParams;

// ---------------------------------------------------------------------------
// ChatSession
// ---------------------------------------------------------------------------
static bool testChatSessionMessagesAndCounters() {
    ChatSession session;
    MINI_LLAMA_ASSERT_EQ(session.getMessages().size(), static_cast<size_t>(0));

    session.addMessage("system", "you are helpful");
    session.addMessage("user", "hi");
    session.addMessage("assistant", "hello");
    MINI_LLAMA_ASSERT_EQ(session.getMessages().size(), static_cast<size_t>(3));
    MINI_LLAMA_ASSERT_EQ(session.getMessages()[0].role, std::string("system"));
    MINI_LLAMA_ASSERT_EQ(session.getMessages()[1].content, std::string("hi"));
    MINI_LLAMA_ASSERT_EQ(session.getMessages()[2].role,
                         std::string("assistant"));

    session.recordTurn(10, 5, 123.0);
    session.recordTurn(2, 3, 7.0);
    MINI_LLAMA_ASSERT_EQ(session.getTotalPromptTokens(), 12);
    MINI_LLAMA_ASSERT_EQ(session.getTotalGeneratedTokens(), 8);
    MINI_LLAMA_ASSERT_NEAR(session.getTotalTime(), 130.0, 1e-9);

    session.appendToken(1);
    session.appendToken(2);
    MINI_LLAMA_ASSERT_EQ(session.getTokenHistory().size(),
                         static_cast<size_t>(2));
    session.setTokenHistory({9, 8, 7});
    MINI_LLAMA_ASSERT_EQ(session.getTokenHistory().size(),
                         static_cast<size_t>(3));
    MINI_LLAMA_ASSERT_EQ(session.getTokenHistory()[0], 9);
    MINI_LLAMA_ASSERT_EQ(session.getTokenHistory()[2], 7);

    SamplingParams params;
    params.temperature = 0.7f;
    params.top_k = 5;
    params.seed = 42;
    session.setSamplingParams(params);
    MINI_LLAMA_ASSERT_NEAR(session.getSamplingParams().temperature, 0.7f,
                           1e-6f);
    MINI_LLAMA_ASSERT_EQ(session.getSamplingParams().top_k, 5);
    MINI_LLAMA_ASSERT_EQ(session.getSamplingParams().seed, 42u);

    std::vector<ChatMessage> replaced = {{"user", "new"}};
    session.setMessages(replaced);
    MINI_LLAMA_ASSERT_EQ(session.getMessages().size(), static_cast<size_t>(1));
    MINI_LLAMA_ASSERT_EQ(session.getMessages()[0].content, std::string("new"));

    // clear 重置全部状态
    session.clear();
    MINI_LLAMA_ASSERT_EQ(session.getMessages().size(), static_cast<size_t>(0));
    MINI_LLAMA_ASSERT_EQ(session.getTokenHistory().size(),
                         static_cast<size_t>(0));
    MINI_LLAMA_ASSERT_EQ(session.getTotalPromptTokens(), 0);
    MINI_LLAMA_ASSERT_EQ(session.getTotalGeneratedTokens(), 0);
    MINI_LLAMA_ASSERT_NEAR(session.getTotalTime(), 0.0, 1e-9);
    MINI_LLAMA_ASSERT_TRUE(session.getPrefixCache().empty());
    return true;
}

static bool testChatSessionPrefixCache() {
    ChatSession session;
    MINI_LLAMA_ASSERT_EQ(session.longestCachedPrefix({1, 2, 3}),
                         static_cast<size_t>(0));

    session.recordPrefix({1, 2, 3});
    MINI_LLAMA_ASSERT_EQ(session.getPrefixCache().size(),
                         static_cast<size_t>(1));
    MINI_LLAMA_ASSERT_TRUE(!session.getPrefixCache().empty());

    // 查询序列比缓存短 / 等长 / 更长 / 完全不匹配
    MINI_LLAMA_ASSERT_EQ(session.longestCachedPrefix({1, 2}),
                         static_cast<size_t>(2));
    MINI_LLAMA_ASSERT_EQ(session.longestCachedPrefix({1, 2, 3}),
                         static_cast<size_t>(3));
    MINI_LLAMA_ASSERT_EQ(session.longestCachedPrefix({1, 2, 3, 4}),
                         static_cast<size_t>(3));
    MINI_LLAMA_ASSERT_EQ(session.longestCachedPrefix({9, 9}),
                         static_cast<size_t>(0));

    // 共享前缀 + 分叉：size 以"序列条数"计
    session.recordPrefix({1, 2, 4});
    MINI_LLAMA_ASSERT_EQ(session.getPrefixCache().size(),
                         static_cast<size_t>(2));
    MINI_LLAMA_ASSERT_EQ(session.longestCachedPrefix({1, 2, 4, 5}),
                         static_cast<size_t>(3));

    session.clear();
    MINI_LLAMA_ASSERT_EQ(session.getPrefixCache().size(),
                         static_cast<size_t>(0));
    MINI_LLAMA_ASSERT_EQ(session.longestCachedPrefix({1, 2, 3}),
                         static_cast<size_t>(0));
    return true;
}

// ---------------------------------------------------------------------------
// PromptBuilder
// ---------------------------------------------------------------------------
static bool testPromptBuilderPlain() {
    PromptBuilder builder;

    std::vector<ChatMessage> single = {{"user", "hi"}};
    MINI_LLAMA_ASSERT_EQ(builder.build(single),
                         std::string("User: hi\nAssistant:"));

    std::vector<ChatMessage> full = {
        {"system", "sys"}, {"user", "hi"}, {"assistant", "hello"}};
    MINI_LLAMA_ASSERT_EQ(
        builder.build(full),
        std::string("System: sys\nUser: hi\nAssistant: hello\nAssistant:"));

    std::vector<ChatMessage> empty;
    MINI_LLAMA_ASSERT_EQ(builder.build(empty), std::string("Assistant:"));

    // 未知模板名回退到 plain 模式
    builder.setChatTemplate("unknown-template");
    MINI_LLAMA_ASSERT_EQ(builder.build(single),
                         std::string("User: hi\nAssistant:"));
    return true;
}

static bool testPromptBuilderQwen2() {
    PromptBuilder builder;
    builder.setChatTemplate("qwen2");

    std::vector<ChatMessage> no_system = {{"user", "hi"}};
    std::string expected =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n";
    MINI_LLAMA_ASSERT_EQ(builder.build(no_system), expected);

    std::vector<ChatMessage> with_system = {{"system", "custom"},
                                            {"user", "hi"}};
    expected =
        "<|im_start|>system\ncustom<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n";
    MINI_LLAMA_ASSERT_EQ(builder.build(with_system), expected);

    // 未支持的 role 在模板输出中被跳过
    std::vector<ChatMessage> unknown_role = {{"tool", "x"}, {"user", "hi"}};
    MINI_LLAMA_ASSERT_EQ(
        builder.build(unknown_role),
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n");
    return true;
}

static bool testPromptBuilderCustomJinjaTemplate() {
    PromptBuilder builder;
    builder.setChatTemplate(
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}");

    std::vector<ChatMessage> messages = {{"user", "hi"}, {"assistant", "yo"}};
    MINI_LLAMA_ASSERT_EQ(builder.build(messages),
                         std::string("user: hi\nassistant: yo\n"));

    std::vector<ChatMessage> empty;
    MINI_LLAMA_ASSERT_EQ(builder.build(empty), std::string(""));
    return true;
}

static bool testLoadChatTemplateFromGguf() {
    // 不存在的文件：返回空串
    MINI_LLAMA_ASSERT_EQ(
        mini_llama::loadChatTemplateFromGguf("models/tiny/does_not_exist.gguf"),
        std::string(""));

    const std::string qwen_path = "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
    if (!std::filesystem::exists(qwen_path)) {
        std::cout << "  (skip: " << qwen_path << " not present)\n";
        return true;
    }

    // 存在的 Qwen2 GGUF：应返回内置 "qwen2" 或包含 <|im_start|> 的原始模板
    std::string tmpl = mini_llama::loadChatTemplateFromGguf(qwen_path);
    MINI_LLAMA_ASSERT_TRUE(!tmpl.empty());
    MINI_LLAMA_ASSERT_TRUE(tmpl == "qwen2" ||
                           tmpl.find("<|im_start|>") != std::string::npos);
    return true;
}

// ---------------------------------------------------------------------------
// 注册
// ---------------------------------------------------------------------------
static struct ChatTestRegistrar {
    ChatTestRegistrar() {
        registerTest("chat_session_messages_and_counters",
                     testChatSessionMessagesAndCounters);
        registerTest("chat_session_prefix_cache", testChatSessionPrefixCache);
        registerTest("prompt_builder_plain", testPromptBuilderPlain);
        registerTest("prompt_builder_qwen2", testPromptBuilderQwen2);
        registerTest("prompt_builder_custom_jinja_template",
                     testPromptBuilderCustomJinjaTemplate);
        registerTest("load_chat_template_from_gguf",
                     testLoadChatTemplateFromGguf);
    }
} chat_test_registrar;
