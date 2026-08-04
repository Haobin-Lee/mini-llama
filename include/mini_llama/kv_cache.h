// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_KV_CACHE_H_
#define INCLUDE_MINI_LLAMA_KV_CACHE_H_

#include "mini_llama/tensor.h"

namespace mini_llama {

// KV cache for all layers and positions.
// keys/values: [n_layers, max_seq_len, n_kv_heads, head_dim]
struct KvCache {
    Tensor keys;
    Tensor values;

    KvCache() = default;
    KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);

    // Write k,v for a (layer,pos). k,v: [n_kv_heads, head_dim].
    void write(int layer, int pos, const Tensor& k, const Tensor& v);

    // Pointers into the cache for a specific (layer,pos,kv_head).
    const float* keyPtr(int layer, int pos, int kv_head) const;
    const float* valuePtr(int layer, int pos, int kv_head) const;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_KV_CACHE_H_
