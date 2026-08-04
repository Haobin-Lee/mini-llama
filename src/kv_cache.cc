// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// TODO(you): implement the KV cache read/write with bounds checks.

#include "mini_llama/kv_cache.h"

#include <stdexcept>

#include "errlog/bizlog.h"

namespace mini_llama {

using errlog::ErrorCode;

static bool isTensor4D(const Tensor& t, const char* name) {
    if (t.numDims() != 4) {
        BIZLOG(ErrorCode::kRequreTensor4D, name);
        return false;
    }

    return true;
}

KvCache::KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
    keys = makeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
    values = makeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
}

void KvCache::write(int layer, int pos, const Tensor& k, const Tensor& v) {
    if (!isTensor4D(keys, "keys") || !isTensor4D(values, "values")) {
        return;
    }

    if (k.numDims() != 2 || v.numDims() != 2 || k.shape != v.shape) {
        BIZLOG(ErrorCode::kInvalidTensorDim,
               "Write expected matching [n_kv_heads, head_dim] tensors");
        return;
    }

    int n_kv_heads = k.shape[0];
    int head_dim = k.shape[1];

    if (layer < 0 || layer >= keys.shape[0]) {
        BIZLOG(ErrorCode::kTensorOutofRange, "Write layer out of range.");
        return;
    }

    if (pos < 0 || pos >= keys.shape[1]) {
        BIZLOG(ErrorCode::kTensorOutofRange, "Write pos out of range.");
        return;
    }

    if (n_kv_heads != keys.shape[2] || head_dim != keys.shape[3]) {
        BIZLOG(ErrorCode::kInvalidTensorDim,
               "Write tensor shape does not match cache shape");
        return;
    }

    size_t layer_stride =
        static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
    size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
    size_t kv_head_stride = static_cast<size_t>(keys.shape[3]);

    size_t base = layer_stride * layer + pos_stride * pos;
    for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
        size_t offset = base + kv_head * kv_head_stride;
        memcpy(&keys.data[offset], &k.data[kv_head * head_dim],
               head_dim * sizeof(float));
        memcpy(&values.data[offset], &v.data[kv_head * head_dim],
               head_dim * sizeof(float));
    }
}

const float* KvCache::keyPtr(int layer, int pos, int kv_head) const {
    if (!isTensor4D(keys, "keys")) {
        return nullptr;
    }
    if (layer < 0 || layer >= keys.shape[0] || pos < 0 ||
        pos >= keys.shape[1] || kv_head < 0 || kv_head >= keys.shape[2]) {
        BIZLOG(ErrorCode::kTensorOutofRange, "KeyPtr flatIndex out of range.");
        return nullptr;
    }

    size_t layer_stride =
        static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
    size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
    size_t kv_head_stride = static_cast<size_t>(keys.shape[3]);

    return &keys.data[layer_stride * layer + pos_stride * pos +
                      kv_head * kv_head_stride];
}
const float* KvCache::valuePtr(int layer, int pos, int kv_head) const {
    if (!isTensor4D(values, "values")) {
        return nullptr;
    }
    if (layer < 0 || layer >= values.shape[0] || pos < 0 ||
        pos >= values.shape[1] || kv_head < 0 || kv_head >= values.shape[2]) {
        BIZLOG(ErrorCode::kTensorOutofRange,
               "ValuePtr flatIndex out of range.");
        return nullptr;
    }

    size_t layer_stride = static_cast<size_t>(
        values.shape[1] * values.shape[2] * values.shape[3]);
    size_t pos_stride = static_cast<size_t>(values.shape[2] * values.shape[3]);
    size_t kv_head_stride = static_cast<size_t>(values.shape[3]);

    return &values.data[layer_stride * layer + pos_stride * pos +
                        kv_head * kv_head_stride];
}

}  // namespace mini_llama
