// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_OPS_H_
#define INCLUDE_MINI_LLAMA_OPS_H_

#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// Matrix multiplication: c = a * b.  a:[m,k] b:[k,n] -> c:[m,n]
Tensor matmul(const Tensor& a, const Tensor& b);

// Linear projection: y = x * weight^T.
// x:[in] or [1,in], weight:[out,in] -> [out] or [1,out] (keeps rank of x).
Tensor linear(const Tensor& x, const Tensor& weight);

// Linear projection with quantized weight.
// Dispatches to F32, Q8_0, or Q4_0 path based on weight.type.
Tensor linear(const Tensor& x, const QuantizedTensor& weight);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight.
Tensor rmsNorm(const Tensor& x, const Tensor& weight, float eps);

// Softmax over a 1D tensor.
Tensor softmax(const Tensor& x);

// SiLU: x * sigmoid(x).
Tensor silu(const Tensor& x);

// Element-wise multiply.
Tensor elementwiseMul(const Tensor& a, const Tensor& b);

// SwiGLU: silu(gate) * up.
Tensor swiGlu(const Tensor& gate, const Tensor& up);

// Rotary position embedding applied in place to q and k.
// q:[n_heads, head_dim], k:[n_kv_heads, head_dim].
bool rope(Tensor& q, Tensor& k, int pos, float theta,
          RopeType rope_type = RopeType::kNormal);

// Index of the maximum value.
int argMax(const Tensor& x);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_OPS_H_
