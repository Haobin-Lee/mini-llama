// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/threadpool.h"

namespace mini_llama {

namespace {
    size_t global_thread_count = 0;
    constexpr size_t kMinChunkSize = 16;
}  // namespace

size_t ThreadPool::getThreadCount() {
    if (global_thread_count == 0) {
        // 初始化线程数
        size_t thread_count = std::thread::hardware_concurrency();
        return thread_count > 0 ? thread_count : 4;
    }
    return global_thread_count;
}

void ThreadPool::setThreadCount(size_t thread_count) {
    global_thread_count = thread_count > 0 ? thread_count : 0;
}

void ThreadPool::submitTask(int n, const Func& func) {
    if (n <= 0) {
        return;
    }

    int n_threads = getThreadCount();
    if (n < kMinChunkSize * n_threads) {
        func(0, n);
        return;
    }

    if (n_threads > n) {
        n_threads = n;
    }

    size_t start = 0;
    size_t avg_chunk_size = n / n_threads;
    size_t reminder = n % n_threads;
    std::vector<std::thread> threads(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        size_t chunk_size = avg_chunk_size + (i < reminder ? 1 : 0);
        size_t end = start + chunk_size;
        threads.emplace_back([&func, start, end]() { func(start, end); });
        start = end;
    }

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

}  // namespace mini_llama
