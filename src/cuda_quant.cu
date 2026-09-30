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

// warp 协作 GEMV
constexpr int kQ80WarpsPerBlock = 4;

// Q8_0 linear: y[row, out] = sum_k(W[out, k] * x[row, k])
// W 以 BlockQ80 存储：每 block = {fp16_scale, int8[32]}
__global__ void q80LinearKernel(const float* __restrict__ x,
                                const BlockQ80* __restrict__ weight,
                                float* __restrict__ y, int rows,
                                int in_features, int out_features,
                                int blocks_per_row) {
  const int lane = threadIdx.x & 31; // warp里的lane索引
  const int warp = threadIdx.x >> 5; // warp索引
  const int row = blockIdx.y; // block所在行
  const int out = blockIdx.x * kQ80WarpsPerBlock + warp; // 每个warp负责计算4个Q80block元素，每个warp中的lane负责计算4个元素
  if (out >= out_features || row >= rows) {
    return;
  }

  const float* x_row = x + static_cast<size_t>(row) * in_features;
  const BlockQ80* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
  const int group = lane >> 3;  // 将lane分为4组，每组8个lane，group编号为0~3
  const int sub = lane & 7;     // 组内编号 0~7

  float sum = 0.0f;
  int k = 0;
  // 主循环：整 128 权重块，权重读取跨 lane 合并。
  for (; k + 4 * kQ80BlockSize <= in_features; k += 4 * kQ80BlockSize) {
    const BlockQ80& block = w_row[(k / kQ80BlockSize) + group];
    // 按字节读取：qs 起点为 2+34*i，4 字节读不满足自然对齐，会触发
    // misaligned address；单字节读天然对齐，且跨 lane 连续可合并。
    const int8_t* q = block.qs + 4 * sub;
    const int base = k + group * kQ80BlockSize + 4 * sub;
    const float d = halfBitsToFloat(block.d);
    sum += d * (static_cast<float>(q[0]) * x_row[base] +
                static_cast<float>(q[1]) * x_row[base + 1] +
                static_cast<float>(q[2]) * x_row[base + 2] +
                static_cast<float>(q[3]) * x_row[base + 3]);
  }
  // 行尾：不足 128 权重的整 block（含最后一个不满 32 的 block）。
  for (; k < in_features; k += kQ80BlockSize) {
    if (k + lane < in_features) {
      const BlockQ80& block = w_row[k / kQ80BlockSize];
      sum += halfBitsToFloat(block.d) *
             static_cast<float>(block.qs[lane]) * x_row[k + lane];
    }
  }

  // warp 内归约，lane 0 写出该行结果。
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) {
    y[static_cast<size_t>(row) * out_features + out] = sum;
  }
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

// Q8_0 GEMV 每 warp 一行，grid.x 按行数除以每 block 的 warp 数换算。
dim3 q80Grid(int out_features, int rows) {
  return dim3((out_features + kQ80WarpsPerBlock - 1) / kQ80WarpsPerBlock,
              rows);
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

  q80LinearKernel<<<q80Grid(p.out_features, p.rows), quantBlock()>>>(
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

  q80LinearKernel<<<q80Grid(p.out_features, p.rows), quantBlock()>>>(
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