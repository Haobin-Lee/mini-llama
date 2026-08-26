// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef MINI_LLAMA_CUDA_MATMUL_H_
#define MINI_LLAMA_CUDA_MATMUL_H_

#include <vector>

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// Returns true if CUDA matmul was compiled in.
bool cudaMatmulBuilt();

// F32 matrix multiplication: c = a @ b.
// a: [m, k], b: [k, n] -> result: [m, n]
Tensor cudaMatmul(const Tensor& a, const Tensor& b, int device_id = 0);

// F32 linear projection: y = x @ weight^T + bias.
// x: [in_features] or [batch, in_features]
// weight: [out_features, in_features]
// bias: optional [out_features]
// result: [out_features] or [batch, out_features]
Tensor cudaLinear(const Tensor& x, const Tensor& weight,
                  const Tensor* bias = nullptr, int device_id = 0);

// Linear with weight already on device (avoids re-uploading weight).
// Input is on host, output is on host.
Tensor cudaLinearDeviceWeight(const Tensor& x, const void* w_device,
                              const std::vector<int>& w_shape,
                              const Tensor* bias = nullptr, int device_id = 0);

// Linear with both input and output on device (zero-copy pipeline).
CudaTensor cudaLinearDeviceInput(const CudaTensor& x, const void* w_device,
                                 const std::vector<int>& w_shape,
                                 int device_id = 0);

}  // namespace mini_llama

#endif  // MINI_LLAMA_CUDA_MATMUL_H_