// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
#define INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_

#include "mini_llama/tensor.h"

namespace mini_llama {

enum class MatmulMode {
    kNaive,
    kThreaded,
    kSimd,
    kThreadedSimd,
};

// Default mode used by the main inference path.
MatmulMode defaultMatmulMode();

// F32 matrix multiplication [m,k] x [k,n] -> [m,n]
Tensor matmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// F32 Linear: x @ weight^T
// Supports x: [in_features] or [1, in_features]
// weight: [out_features, in_features]
// result rank matches input rank.
Tensor linearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
