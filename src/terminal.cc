// Copyright (c) 2026
// SPDX-License-Identifier: MIT

// implement terminal I/O helpers.

#include "mini_llama/terminal.h"

#include <iomanip>
#include <iostream>
#include <string>

namespace mini_llama {

void Terminal::printUserPrompt() const {
    std::cout << "mini-llama > ";
    std::cout.flush();
}

std::string Terminal::readLine() const {
    std::string line;
    if (std::getline(std::cin, line)) {
        return line;
    }
    return "";
}

void Terminal::printAssistantPrefix() const {
    std::cout << "assistant  > ";
    std::cout.flush();
}

// 输出模型生成的文本
void Terminal::printTokenText(const std::string& text) const {
    std::cout << text;
}

// 刷新输出缓冲区
void Terminal::flush() const { std::cout.flush(); }

void Terminal::newLine() const { std::cout << "\n"; }

// 输出帮助信息
void Terminal::printHelp() const {
    std::cout << "Commands:\n"
              << "  /help    Show this help message\n"
              << "  /clear   Clear chat history and context\n"
              << "  /stats   Show session statistics\n"
              << "  /params  Show current sampling parameters\n"
              << "  /exit    Exit chat\n";
}

void Terminal::printStats(const ChatSession& session) const {
    std::cout << "Session stats:\n"
              << "  messages:           " << session.getMessages().size()
              << "\n"
              << "  context tokens:     " << session.getTokenHistory().size()
              << "\n"
              << "  total prompt tokens: " << session.getTotalPromptTokens()
              << "\n"
              << "  total generated:     " << session.getTotalGeneratedTokens()
              << "\n"
              << "  total time:          " << std::fixed << std::setprecision(2)
              << session.getTotalTime() << " ms\n"
              << "  tokens/s:            ";
    if (session.getTotalTime() > 0.0) {
        std::cout << std::fixed << std::setprecision(2)
                  << (session.getTotalGeneratedTokens() * 1000.0 /
                      session.getTotalTime());
    } else {
        std::cout << "n/a";
    }
    std::cout << std::defaultfloat << "\n";
}

// 输出采样参数
void Terminal::printParams(const SamplingParams& params) const {
    std::cout << std::defaultfloat << "Sampling params:\n"
              << "  temperature: " << params.temperature << "\n"
              << "  top_k:       " << params.top_k << "\n"
              << "  seed:        " << params.seed << "\n";
}

void Terminal::printMessage(const std::string& msg) const {
    std::cout << msg << "\n";
}

}  // namespace mini_llama