// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_THREADPOOL_H_
#define INCLUDE_MINI_LLAMA_THREADPOOL_H_

#include <functional>
#include <thread>

namespace mini_llama {

class ThreadPool {
    using Func = std::function<void(size_t start, size_t end)>;

public:
    static size_t getThreadCount();
    static void setThreadCount(size_t thread_count);
    static void submitTask(int head, const Func& func);
};

}  // namespace mini_llama
#endif  // INCLUDE_MINI_LLAMA_THREADPOOL_H_
