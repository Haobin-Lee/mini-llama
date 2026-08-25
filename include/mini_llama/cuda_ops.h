// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef MINI_LLAMA_CUDA_OPS_H_
#define MINI_LLAMA_CUDA_OPS_H_

#include <vector>

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// Returns true if CUDA ops were compiled in.
bool cudaOpsBuilt();

// ---------------------------------------------------------------------------
// Host interface: Tensor in, Tensor out (Upload → Kernel → Download)
// ---------------------------------------------------------------------------

// RMS normalization: y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
Tensor cudaRmsNorm(const Tensor& x, const Tensor& weight, float eps,
                   int device_id = 0);

// SiLU activation: y[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
Tensor cudaSilu(const Tensor& x, int device_id = 0);

// Element-wise multiplication: y[i] = a[i] * b[i]
Tensor cudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id = 0);

// Element-wise addition: y[i] = a[i] + b[i]
Tensor cudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id = 0);

// ---------------------------------------------------------------------------
// Device interface: CudaTensor in, CudaTensor out (zero-copy on GPU)
// ---------------------------------------------------------------------------

// RMS norm with device input and device weight (weight is a raw device ptr).
CudaTensor cudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   float eps, int device_id = 0);

// RMS norm with device input and host weight (uploads weight once).
CudaTensor cudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight,
                                  float eps, int device_id = 0);

// SiLU with device input.
CudaTensor cudaSiluDeviceInput(const CudaTensor& x, int device_id = 0);

// Element-wise mul with device inputs.
CudaTensor cudaElementwiseMulDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b,
                                         int device_id = 0);

// Element-wise add with device inputs.
CudaTensor cudaElementwiseAddDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b,
                                         int device_id = 0);

// Element-wise add with device input + device weight (weight is a raw ptr).
CudaTensor cudaElementwiseAddDeviceWeight(const CudaTensor& a,
                                          const void* b_data,
                                          const std::vector<int>& b_shape,
                                          int device_id = 0);

// Softmax: y[i] = exp(x[i] - max) / sum(exp(x[j] - max))
Tensor cudaSoftmax(const Tensor& x, int device_id = 0);

// Embedding lookup: y = embedding[token_id, :], weight pre-uploaded on device.
CudaTensor cudaEmbeddingLookupDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    int token_id, int device_id = 0);

// RoPE: applies rotary position encoding to q and k in-place.
// q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
void cudaRope(Tensor& q, Tensor& k, int pos, float theta,
              RopeType rope_type = RopeType::kNormal, int device_id = 0);

// RoPE with device tensors (in-place, zero-copy).
void cudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                         int n_kv_heads, int head_dim, int pos, float theta,
                         RopeType rope_type = RopeType::kNormal,
                         int device_id = 0);
}  // namespace mini_llama

#endif  // MINI_LLAMA_CUDA_OPS_H_
