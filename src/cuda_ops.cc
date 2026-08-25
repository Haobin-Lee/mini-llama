// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/cuda_ops.h"

#include <stdexcept>
#include <vector>

namespace mini_llama {
namespace {

    std::runtime_error cudaOpsNotBuiltError() {
        return std::runtime_error(
            "CUDA ops were not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on "
            "a "
            "NVIDIA CUDA machine.");
    }

}  // namespace

bool cudaOpsBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

// cpu wrong path
#ifndef MINI_LLAMA_USE_CUDA

Tensor cudaRmsNorm(const Tensor& x, const Tensor& weight, float eps,
                   int device_id) {
    (void)x;
    (void)weight;
    (void)eps;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

Tensor cudaSilu(const Tensor& x, int device_id) {
    (void)x;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

Tensor cudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

Tensor cudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

CudaTensor cudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   float eps, int device_id) {
    (void)x;
    (void)weight_data;
    (void)weight_shape;
    (void)eps;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

CudaTensor cudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight,
                                  float eps, int device_id) {
    (void)x;
    (void)weight;
    (void)eps;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

CudaTensor cudaSiluDeviceInput(const CudaTensor& x, int device_id) {
    (void)x;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

CudaTensor cudaElementwiseMulDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

CudaTensor cudaElementwiseAddDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b, int device_id) {
    (void)a;
    (void)b;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

CudaTensor cudaElementwiseAddDeviceWeight(const CudaTensor& a,
                                          const void* b_data,
                                          const std::vector<int>& b_shape,
                                          int device_id) {
    (void)a;
    (void)b_data;
    (void)b_shape;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

// ---------------------------------------------------------------------------
// Softmax, Embedding, RoPE
// ---------------------------------------------------------------------------

Tensor cudaSoftmax(const Tensor& x, int device_id) {
    (void)x;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

CudaTensor cudaEmbeddingLookupDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    int token_id, int device_id) {
    (void)embedding_data;
    (void)embedding_shape;
    (void)token_id;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

void cudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type,
              int device_id) {
    (void)q;
    (void)k;
    (void)pos;
    (void)theta;
    (void)rope_type;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

void cudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                         int n_kv_heads, int head_dim, int pos, float theta,
                         RopeType rope_type, int device_id) {
    (void)q;
    (void)k;
    (void)n_heads;
    (void)n_kv_heads;
    (void)head_dim;
    (void)pos;
    (void)theta;
    (void)rope_type;
    (void)device_id;
    throw cudaOpsNotBuiltError();
}

#endif  // MINI_LLAMA_USE_CUDA

}  // namespace mini_llama