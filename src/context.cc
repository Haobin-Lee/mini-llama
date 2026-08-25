// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// implement the context constructor (allocate the KV cache from the model
// config).

#include "mini_llama/context.h"

#include <stdexcept>

namespace mini_llama {

MiniLlamaContext::MiniLlamaContext(const MiniLlamaModel* model) {
    if (model) {
        const auto& c = model->config;
        kv_cache = KvCache(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim);
        if (model->cuda_weights) {
            cuda_kv_cache =
                CudaKvCache(c.n_layers, c.max_seq_len, c.n_kv_heads,
                            +c.head_dim, model->cuda_weights->device_id);
        }
    }
}

}  // namespace mini_llama
