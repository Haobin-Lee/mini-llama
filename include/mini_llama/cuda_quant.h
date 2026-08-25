#ifndef MINI_LLAMA_CUDA_QUANT_H_
#define MINI_LLAMA_CUDA_QUANT_H_

#include <cstddef>
#include <vector>

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool cudaQuantBuilt();

// ---------------------------------------------------------------------------
// Q8_0 quantized linear: y = x @ W^T + bias
// ---------------------------------------------------------------------------

// Host weight, host input, host output.
Tensor cudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

// Device weight (pre-uploaded), host input, host output.
Tensor cudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device,
                                 size_t q8_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr,
                                 int device_id = 0);

// Device weight, device input, device output (zero-copy pipeline).
CudaTensor cudaQ80LinearDeviceInput(const CudaTensor& x,
                                    const void* q8_0_device,
                                    size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

// ---------------------------------------------------------------------------
// Q4_0 quantized linear
// ---------------------------------------------------------------------------

Tensor cudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

Tensor cudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device,
                                 size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr,
                                 int device_id = 0);

CudaTensor cudaQ40LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_0_device,
                                    size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

// ---------------------------------------------------------------------------
// Q4_1 quantized linear
// ---------------------------------------------------------------------------

Tensor cudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

Tensor cudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device,
                                 size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr,
                                 int device_id = 0);

CudaTensor cudaQ41LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_1_device,
                                    size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

}  // namespace mini_llama

#endif  // MINI_LLAMA_CUDA_QUANT_H_