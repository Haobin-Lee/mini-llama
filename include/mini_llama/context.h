// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CONTEXT_H_
#define INCLUDE_MINI_LLAMA_CONTEXT_H_

#include <vector>

#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"

namespace mini_llama {

// Inference state: KV cache, current position, token history, stats.
struct MiniLlamaContext {
    const MiniLlamaModel* model = nullptr;
    KvCache kv_cache;
    int pos = 0;

    std::vector<int> token_history;

    int n_prefill_tokens = 0;
    int n_decode_tokens = 0;

    MiniLlamaContext() = default;
    MiniLlamaContext(const MiniLlamaContext&) = delete;
    MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;
    MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
    MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;
    explicit MiniLlamaContext(const MiniLlamaModel* model);
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CONTEXT_H_
