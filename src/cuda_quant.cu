// clang-format off
#include "mini_llama/cuda_quant.h"
// clang-format on

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"

namespace mini_llama {
namespace {

constexpr int kBlockSize = 128;

void checkCudaQuant(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA quant error in " + std::string(expr) +
                             ": " + cudaGetErrorString(err));
  }
}

void checkLastKernel(const char* name) {
  checkCudaQuant(cudaGetLastError(), name);
}

// ---------------------------------------------------------------------------
// Device helpers: FP16 bits → float
// ---------------------------------------------------------------------------

// Converts a raw uint16_t (IEEE 754 half-precision bit pattern) to float.
// Used to decode the fp16 scale/min fields in quantized blocks.
__device__ float halfBitsToFloat(uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return __half2float(raw);
}

// ---------------------------------------------------------------------------
// Q4_0 and Q4_1 dequantization helpers (device functions)
// ---------------------------------------------------------------------------

// Q4_0: dequantize one element from a block.
// Block stores 32 4-bit values packed into qs[16] (2 per byte).
// flat_index is in [0, 31]. Value = d * (q - 8).
__device__ float q40DequantValue(const BlockQ40& block, int flat_index) {
  // Lower nibble (index 0-15) or upper nibble (index 16-31)
  int q = flat_index < 16
              ? static_cast<int>(block.qs[flat_index] & 0x0F)
              : static_cast<int>(block.qs[flat_index - 16] >> 4);
  return halfBitsToFloat(block.d) * static_cast<float>(q - 8);
}

// Q4_1: dequantize one element from a block.
// Value = d * q + m. The extra 'm' offset allows asymmetric ranges.
__device__ float q41DequantValue(const BlockQ41& block, int flat_index) {
  int q = flat_index < 16
              ? static_cast<int>(block.qs[flat_index] & 0x0F)
              : static_cast<int>(block.qs[flat_index - 16] >> 4);
  return halfBitsToFloat(block.d) * static_cast<float>(q) +
         halfBitsToFloat(block.m);
}

// ===========================================================================
// CUDA Kernels
// ===========================================================================
//
// All three kernels share the same structure:
//   - Each thread computes ONE output element: y[row, out]
//   - blockIdx.y = row (which input row)
//   - blockIdx.x * blockDim.x + threadIdx.x = out (which output feature)
//   - The thread loops over all quantized blocks in that weight row,
//     dequantizing on-the-fly and accumulating the dot product.
//
// This is a "dequantize-then-multiply" approach: we never write the
// dequantized weight back to global memory, saving bandwidth.

// Q8_0 linear: y[row, out] = sum_k(W[out, k] * x[row, k])
// W is stored as BlockQ80: each block = {fp16_scale, int8[32]}
__global__ void q80LinearKernel(const float* x, const BlockQ80* weight,
                                float* y, int rows, int in_features,
                                int out_features, int blocks_per_row) {
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (out >= out_features || row >= rows) {
    return;
  }

  const float* x_row = x + static_cast<size_t>(row) * in_features;
  const BlockQ80* w_row = weight + static_cast<size_t>(out) * blocks_per_row;

  float sum = 0.0f;
  for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const BlockQ80& block = w_row[block_idx];
    float d = halfBitsToFloat(block.d);
    int base_k = block_idx * kQ80BlockSize;
    int k_end = base_k + kQ80BlockSize < in_features
                    ? base_k + kQ80BlockSize
                    : in_features;
    for (int k = base_k; k < k_end; ++k) {
      sum += d * static_cast<float>(block.qs[k - base_k]) * x_row[k];
    }
  }
  y[static_cast<size_t>(row) * out_features + out] = sum;
}

// Q4_0 linear: same structure, uses q40DequantValue for dequantization.
__global__ void q40LinearKernel(const float* x, const BlockQ40* weight,
                                float* y, int rows, int in_features,
                                int out_features, int blocks_per_row) {
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (out >= out_features || row >= rows) {
    return;
  }

  const float* x_row = x + static_cast<size_t>(row) * in_features;
  const BlockQ40* w_row = weight + static_cast<size_t>(out) * blocks_per_row;

  float sum = 0.0f;
  for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const BlockQ40& block = w_row[block_idx];
    int base_k = block_idx * kQ40BlockSize;
    int k_end = base_k + kQ40BlockSize < in_features
                    ? base_k + kQ40BlockSize
                    : in_features;
    for (int k = base_k; k < k_end; ++k) {
      sum += q40DequantValue(block, k - base_k) * x_row[k];
    }
  }
  y[static_cast<size_t>(row) * out_features + out] = sum;
}

// Q4_1 linear: same structure, uses q41DequantValue.
__global__ void q41LinearKernel(const float* x, const BlockQ41* weight,
                                float* y, int rows, int in_features,
                                int out_features, int blocks_per_row) {
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (out >= out_features || row >= rows) {
    return;
  }

  const float* x_row = x + static_cast<size_t>(row) * in_features;
  const BlockQ41* w_row = weight + static_cast<size_t>(out) * blocks_per_row;

  float sum = 0.0f;
  for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
    const BlockQ41& block = w_row[block_idx];
    int base_k = block_idx * kQ41BlockSize;
    int k_end = base_k + kQ41BlockSize < in_features
                    ? base_k + kQ41BlockSize
                    : in_features;
    for (int k = base_k; k < k_end; ++k) {
      sum += q41DequantValue(block, k - base_k) * x_row[k];
    }
  }
  y[static_cast<size_t>(row) * out_features + out] = sum;
}

// ---------------------------------------------------------------------------
// Host-side validation helpers
// ---------------------------------------------------------------------------

int quantInFeatures(const std::vector<int>& x_shape,
                    const std::string& x_str, const char* caller) {
  if (x_shape.size() == 1) return x_shape[0];
  if (x_shape.size() == 2) return x_shape[1];
  throw std::runtime_error(
      std::string(caller) +
      ": expected x shape [in_features] or [batch, in_features], got " +
      x_str);
}

void validateQuantLinearShape(const std::vector<int>& x_shape,
                              const std::string& x_str,
                              const std::vector<int>& w_shape,
                              size_t block_count, int block_size,
                              const char* caller) {
  if (w_shape.size() != 2) {
    throw std::runtime_error(
        std::string(caller) +
        ": expected weight shape [out_features, in_features]");
  }
  int in_feat = quantInFeatures(x_shape, x_str, caller);
  if (in_feat <= 0 || w_shape[0] <= 0 || w_shape[1] <= 0) {
    throw std::runtime_error(
        std::string(caller) + ": empty tensors are not supported");
  }
  if (w_shape[1] != in_feat) {
    throw std::runtime_error(std::string(caller) + ": dimension mismatch x=" +
                             x_str + " weight=[" +
                             std::to_string(w_shape[0]) + ", " +
                             std::to_string(w_shape[1]) + "]");
  }
  int blocks_per_row = (in_feat + block_size - 1) / block_size;
  size_t expected = static_cast<size_t>(w_shape[0]) * blocks_per_row;
  if (block_count != expected) {
    throw std::runtime_error(std::string(caller) +
                             ": block count mismatch: expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(block_count));
  }
}

void addBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      y.data[static_cast<size_t>(row) * cols + col] += bias.data[col];
    }
  }
}

// ---------------------------------------------------------------------------
// Shared launch logic
// ---------------------------------------------------------------------------

struct LinearParams {
  bool input_is_1d;
  int rows;
  int in_features;
  int out_features;
  int blocks_per_row;
};

LinearParams computeLinearParams(const std::vector<int>& x_shape,
                                 const std::vector<int>& w_shape,
                                 int block_size) {
  LinearParams p;
  p.input_is_1d = x_shape.size() == 1;
  p.rows = p.input_is_1d ? 1 : x_shape[0];
  p.in_features = p.input_is_1d ? x_shape[0] : x_shape[1];
  p.out_features = w_shape[0];
  p.blocks_per_row =
      (p.in_features + block_size - 1) / block_size;
  return p;
}

dim3 quantGrid(int out_features, int rows) {
  return dim3((out_features + kBlockSize - 1) / kBlockSize, rows);
}

dim3 quantBlock() { return dim3(kBlockSize); }

std::vector<int> outputShape(bool input_is_1d, int rows, int out_features) {
  return input_is_1d ? std::vector<int>{out_features}
                     : std::vector<int>{rows, out_features};
}

}  // namespace

// ===========================================================================
// Public API
// ===========================================================================

// ---------------------------------------------------------------------------
// Q8_0
// ---------------------------------------------------------------------------

Tensor cudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  validateQuantLinearShape(x.shape, x.shapeString(), weight_shape,
                           weight.size(), kQ80BlockSize, "cudaQ80Linear");
  CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ80), device_id);
  w_dev.upload(weight.data(), weight.size() * sizeof(BlockQ80));
  return cudaQ80LinearDeviceWeight(x, w_dev.data(), weight.size(),
                                   weight_shape, bias, device_id);
}

Tensor cudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device,
                                 size_t q8_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  static constexpr const char* kCaller = "cudaQ80Linear";
  if (q8_0_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) + ": null weight pointer");
  }
  validateQuantLinearShape(x.shape, x.shapeString(), weight_shape,
                           q8_0_block_count, kQ80BlockSize, kCaller);

  cudaSetDeviceId(device_id);
  auto p = computeLinearParams(x.shape, weight_shape, kQ80BlockSize);
  Tensor y(outputShape(p.input_is_1d, p.rows, p.out_features), 0.0f);

  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.upload(x.data.data(), x.size() * sizeof(float));

  q80LinearKernel<<<quantGrid(p.out_features, p.rows), quantBlock()>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<const BlockQ80*>(q8_0_device),
      static_cast<float*>(y_dev.data()), p.rows, p.in_features,
      p.out_features, p.blocks_per_row);
  checkLastKernel("q80LinearKernel");

  y_dev.download(y.data.data(), y.size() * sizeof(float));
  if (bias != nullptr) {
    addBiasInPlace(y, *bias, p.rows, p.out_features);
  }
  return y;
}

CudaTensor cudaQ80LinearDeviceInput(const CudaTensor& x,
                                    const void* q8_0_device,
                                    size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  static constexpr const char* kCaller = "cudaQ80LinearDeviceInput";
  if (q8_0_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) + ": null weight pointer");
  }
  if (x.deviceId() != device_id) {
    throw std::runtime_error(
        std::string(kCaller) + ": input tensor on different CUDA device");
  }
  validateQuantLinearShape(x.shape(), x.shapeString(), weight_shape,
                           q8_0_block_count, kQ80BlockSize, kCaller);

  cudaSetDeviceId(device_id);
  auto p = computeLinearParams(x.shape(), weight_shape, kQ80BlockSize);
  CudaTensor y(outputShape(p.input_is_1d, p.rows, p.out_features), device_id);

  q80LinearKernel<<<quantGrid(p.out_features, p.rows), quantBlock()>>>(
      static_cast<const float*>(x.data()),
      static_cast<const BlockQ80*>(q8_0_device),
      static_cast<float*>(y.data()), p.rows, p.in_features, p.out_features,
      p.blocks_per_row);
  checkLastKernel("q80LinearKernel");

  return y;
}

// ---------------------------------------------------------------------------
// Q4_0
// ---------------------------------------------------------------------------

Tensor cudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  validateQuantLinearShape(x.shape, x.shapeString(), weight_shape,
                           weight.size(), kQ40BlockSize, "cudaQ40Linear");
  CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ40), device_id);
  w_dev.upload(weight.data(), weight.size() * sizeof(BlockQ40));
  return cudaQ40LinearDeviceWeight(x, w_dev.data(), weight.size(),
                                   weight_shape, bias, device_id);
}

Tensor cudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device,
                                 size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  static constexpr const char* kCaller = "cudaQ40Linear";
  if (q4_0_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) + ": null weight pointer");
  }
  validateQuantLinearShape(x.shape, x.shapeString(), weight_shape,
                           q4_0_block_count, kQ40BlockSize, kCaller);

  cudaSetDeviceId(device_id);
  auto p = computeLinearParams(x.shape, weight_shape, kQ40BlockSize);
  Tensor y(outputShape(p.input_is_1d, p.rows, p.out_features), 0.0f);

  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.upload(x.data.data(), x.size() * sizeof(float));

  q40LinearKernel<<<quantGrid(p.out_features, p.rows), quantBlock()>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<const BlockQ40*>(q4_0_device),
      static_cast<float*>(y_dev.data()), p.rows, p.in_features,
      p.out_features, p.blocks_per_row);
  checkLastKernel("q40LinearKernel");

  y_dev.download(y.data.data(), y.size() * sizeof(float));
  if (bias != nullptr) {
    addBiasInPlace(y, *bias, p.rows, p.out_features);
  }
  return y;
}

CudaTensor cudaQ40LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_0_device,
                                    size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  static constexpr const char* kCaller = "cudaQ40LinearDeviceInput";
  if (q4_0_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) + ": null weight pointer");
  }
  if (x.deviceId() != device_id) {
    throw std::runtime_error(
        std::string(kCaller) + ": input tensor on different CUDA device");
  }
  validateQuantLinearShape(x.shape(), x.shapeString(), weight_shape,
                           q4_0_block_count, kQ40BlockSize, kCaller);

  cudaSetDeviceId(device_id);
  auto p = computeLinearParams(x.shape(), weight_shape, kQ40BlockSize);
  CudaTensor y(outputShape(p.input_is_1d, p.rows, p.out_features), device_id);

  q40LinearKernel<<<quantGrid(p.out_features, p.rows), quantBlock()>>>(
      static_cast<const float*>(x.data()),
      static_cast<const BlockQ40*>(q4_0_device),
      static_cast<float*>(y.data()), p.rows, p.in_features, p.out_features,
      p.blocks_per_row);
  checkLastKernel("q40LinearKernel");

  return y;
}

// ---------------------------------------------------------------------------
// Q4_1
// ---------------------------------------------------------------------------

Tensor cudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  validateQuantLinearShape(x.shape, x.shapeString(), weight_shape,
                           weight.size(), kQ41BlockSize, "cudaQ41Linear");
  CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ41), device_id);
  w_dev.upload(weight.data(), weight.size() * sizeof(BlockQ41));
  return cudaQ41LinearDeviceWeight(x, w_dev.data(), weight.size(),
                                   weight_shape, bias, device_id);
}

Tensor cudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device,
                                 size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  static constexpr const char* kCaller = "cudaQ41Linear";
  if (q4_1_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) + ": null weight pointer");
  }
  validateQuantLinearShape(x.shape, x.shapeString(), weight_shape,
                           q4_1_block_count, kQ41BlockSize, kCaller);

  cudaSetDeviceId(device_id);
  auto p = computeLinearParams(x.shape, weight_shape, kQ41BlockSize);
  Tensor y(outputShape(p.input_is_1d, p.rows, p.out_features), 0.0f);

  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.upload(x.data.data(), x.size() * sizeof(float));

  q41LinearKernel<<<quantGrid(p.out_features, p.rows), quantBlock()>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<const BlockQ41*>(q4_1_device),
      static_cast<float*>(y_dev.data()), p.rows, p.in_features,
      p.out_features, p.blocks_per_row);
  checkLastKernel("q41LinearKernel");

  y_dev.download(y.data.data(), y.size() * sizeof(float));
  if (bias != nullptr) {
    addBiasInPlace(y, *bias, p.rows, p.out_features);
  }
  return y;
}

CudaTensor cudaQ41LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_1_device,
                                    size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  static constexpr const char* kCaller = "cudaQ41LinearDeviceInput";
  if (q4_1_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) + ": null weight pointer");
  }
  if (x.deviceId() != device_id) {
    throw std::runtime_error(
        std::string(kCaller) + ": input tensor on different CUDA device");
  }
  validateQuantLinearShape(x.shape(), x.shapeString(), weight_shape,
                           q4_1_block_count, kQ41BlockSize, kCaller);

  cudaSetDeviceId(device_id);
  auto p = computeLinearParams(x.shape(), weight_shape, kQ41BlockSize);
  CudaTensor y(outputShape(p.input_is_1d, p.rows, p.out_features), device_id);

  q41LinearKernel<<<quantGrid(p.out_features, p.rows), quantBlock()>>>(
      static_cast<const float*>(x.data()),
      static_cast<const BlockQ41*>(q4_1_device),
      static_cast<float*>(y.data()), p.rows, p.in_features, p.out_features,
      p.blocks_per_row);
  checkLastKernel("q41LinearKernel");

  return y;
}

}  // namespace mini_llama