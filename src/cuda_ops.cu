// Copyright (c) 2026
// SPDX-License-Identifier: MIT

// clang-format off
#include "mini_llama/cuda_ops.h"
// clang-format on

#include <cuda_runtime_api.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"

namespace mini_llama {
namespace {

constexpr int kBlockSize = 256;

int gridFor(int n) { return (n + kBlockSize - 1) / kBlockSize; }

void checkCudaOps(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA ops error in " + std::string(expr) + ": " +
                             cudaGetErrorString(err));
  }
}

void checkLastKernel(const char* name) {
  checkCudaOps(cudaGetLastError(), name);
}

// ---------------------------------------------------------------------------
// Input validation helpers
// ---------------------------------------------------------------------------

void requireSameShape(const Tensor& a, const Tensor& b, const char* caller) {
  if (a.shape != b.shape) {
    throw std::runtime_error(std::string(caller) + ": shape mismatch " +
                             a.shapeString() + " vs " +
                             b.shapeString());
  }
}

void requireSameShape(const CudaTensor& a, const CudaTensor& b,
                      const char* caller) {
  if (a.shape() != b.shape()) {
    throw std::runtime_error(std::string(caller) + ": shape mismatch " +
                             a.shapeString() + " vs " +
                             b.shapeString());
  }
  if (a.deviceId() != b.deviceId()) {
    throw std::runtime_error(std::string(caller) +
                             ": tensors are on different CUDA devices");
  }
}

void require1D(const Tensor& x, const char* caller) {
  if (x.numDims() != 1) {
    throw std::runtime_error(std::string(caller) +
                             ": expected 1D tensor, got " +
                             x.shapeString());
  }
  if (x.size() == 0) {
    throw std::runtime_error(std::string(caller) + ": empty tensor");
  }
}

void require1D(const CudaTensor& x, const char* caller) {
  if (x.numDims() != 1) {
    throw std::runtime_error(std::string(caller) +
                             ": expected 1D tensor, got " +
                             x.shapeString());
  }
  if (x.size() == 0) {
    throw std::runtime_error(std::string(caller) + ": empty tensor");
  }
}

void validateRmsNormDeviceWeight(const CudaTensor& x,
                                 const void* weight_data,
                                 const std::vector<int>& weight_shape,
                                 float eps, int device_id) {
  require1D(x, "cudaRmsNormDeviceWeight");
  if (weight_data == nullptr) {
    throw std::runtime_error("cudaRmsNormDeviceWeight: weight data is null");
  }
  if (x.shape() != weight_shape) {
    throw std::runtime_error(
        "cudaRmsNormDeviceWeight: x and weight shape mismatch");
  }
  if (x.deviceId() != device_id) {
    throw std::runtime_error(
        "cudaRmsNormDeviceWeight: input tensor is on a different CUDA device");
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error(
        "cudaRmsNormDeviceWeight: eps must be finite and positive");
  }
}

void validateAddDeviceWeight(const CudaTensor& a, const void* b_data,
                             const std::vector<int>& b_shape,
                             int device_id) {
  if (a.deviceId() != device_id) {
    throw std::runtime_error(
        "cudaElementwiseAddDeviceWeight: input tensor is on a different CUDA "
        "device");
  }
  if (b_data == nullptr) {
    throw std::runtime_error(
        "cudaElementwiseAddDeviceWeight: weight data is null");
  }
  if (a.shape() != b_shape) {
    throw std::runtime_error(
        "cudaElementwiseAddDeviceWeight: input and weight shape mismatch");
  }
}

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

// Accumulates sum of squares: *sum += x[i]^2 for all i.
// Uses atomicAdd so only a single float of output is needed.
__global__ void sumSquaresKernel(const float* x, float* sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(sum, x[i] * x[i]);
  }
}

// RMS normalization: y[i] = x[i] * rsqrt(mean + eps) * weight[i].
__global__ void rmsNormKernel(const float* x, const float* weight, float* y,
                              const float* sum, int n, float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float scale = rsqrtf((*sum / static_cast<float>(n)) + eps);
    y[i] = x[i] * scale * weight[i];
  }
}

// SiLU activation: y[i] = x[i] / (1 + exp(-x[i])).
__global__ void siluKernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
  }
}

// Element-wise multiplication: y[i] = a[i] * b[i].
__global__ void elementwiseMulKernel(const float* a, const float* b, float* y,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] * b[i];
  }
}

// Element-wise addition: y[i] = a[i] + b[i].
__global__ void elementwiseAddKernel(const float* a, const float* b, float* y,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] + b[i];
  }
}

}  // namespace


// ---------------------------------------------------------------------------
// Host interface
// ---------------------------------------------------------------------------

Tensor cudaRmsNorm(const Tensor& x, const Tensor& weight, float eps,
                   int device_id) {
  require1D(x, "cudaRmsNorm");
  require1D(weight, "cudaRmsNorm");
  if (x.shape != weight.shape) {
    throw std::runtime_error("cudaRmsNorm: x and weight shape mismatch");
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error("cudaRmsNorm: eps must be finite and positive");
  }

  cudaSetDeviceId(device_id);
  Tensor y(x.shape, 0.0f);

  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);

  x_dev.upload(x.data.data(), x.size() * sizeof(float));
  w_dev.upload(weight.data.data(), weight.size() * sizeof(float));
  checkCudaOps(cudaMemset(sum_dev.data(), 0, sizeof(float)),
               "cudaMemset(rmsNorm sum)");

  int n = static_cast<int>(x.size());
  sumSquaresKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<float*>(sum_dev.data()), n);
  checkLastKernel("sumSquaresKernel");

  rmsNormKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<const float*>(w_dev.data()),
      static_cast<float*>(y_dev.data()),
      static_cast<const float*>(sum_dev.data()), n, eps);
  checkLastKernel("rmsNormKernel");

  y_dev.download(y.data.data(), y.size() * sizeof(float));
  return y;
}

Tensor cudaSilu(const Tensor& x, int device_id) {
  cudaSetDeviceId(device_id);
  Tensor y(x.shape, 0.0f);

  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.upload(x.data.data(), x.size() * sizeof(float));

  int n = static_cast<int>(x.size());
  siluKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<float*>(y_dev.data()), n);
  checkLastKernel("siluKernel");

  y_dev.download(y.data.data(), y.size() * sizeof(float));
  return y;
}

Tensor cudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id) {
  requireSameShape(a, b, "cudaElementwiseMul");
  cudaSetDeviceId(device_id);
  Tensor y(a.shape, 0.0f);

  CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
  CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  a_dev.upload(a.data.data(), a.size() * sizeof(float));
  b_dev.upload(b.data.data(), b.size() * sizeof(float));

  int n = static_cast<int>(a.size());
  elementwiseMulKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<const float*>(a_dev.data()),
      static_cast<const float*>(b_dev.data()),
      static_cast<float*>(y_dev.data()), n);
  checkLastKernel("elementwiseMulKernel");

  y_dev.download(y.data.data(), y.size() * sizeof(float));
  return y;
}

Tensor cudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id) {
  requireSameShape(a, b, "cudaElementwiseAdd");
  cudaSetDeviceId(device_id);
  Tensor y(a.shape, 0.0f);

  CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
  CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  a_dev.upload(a.data.data(), a.size() * sizeof(float));
  b_dev.upload(b.data.data(), b.size() * sizeof(float));

  int n = static_cast<int>(a.size());
  elementwiseAddKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<const float*>(a_dev.data()),
      static_cast<const float*>(b_dev.data()),
      static_cast<float*>(y_dev.data()), n);
  checkLastKernel("elementwiseAddKernel");

  y_dev.download(y.data.data(), y.size() * sizeof(float));
  return y;
}

// ---------------------------------------------------------------------------
// Device interface
// ---------------------------------------------------------------------------

CudaTensor cudaRmsNormDeviceWeight(const CudaTensor& x,
                                   const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   float eps, int device_id) {
  validateRmsNormDeviceWeight(x, weight_data, weight_shape, eps, device_id);
  cudaSetDeviceId(device_id);

  CudaTensor y(x.shape(), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);
  checkCudaOps(cudaMemset(sum_dev.data(), 0, sizeof(float)),
               "cudaMemset(rmsNorm sum)");

  int n = static_cast<int>(x.size());
  sumSquaresKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<const float*>(x.data()), static_cast<float*>(sum_dev.data()),
      n);
  checkLastKernel("sumSquaresKernel");

  rmsNormKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<const float*>(x.data()),
      static_cast<const float*>(weight_data), static_cast<float*>(y.data()),
      static_cast<const float*>(sum_dev.data()), n, eps);
  checkLastKernel("rmsNormKernel");

  return y;
}

CudaTensor cudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight,
                                  float eps, int device_id) {
  require1D(x, "cudaRmsNormDeviceInput");
  require1D(weight, "cudaRmsNormDeviceInput");
  if (x.shape() != weight.shape) {
    throw std::runtime_error(
        "cudaRmsNormDeviceInput: x and weight shape mismatch");
  }
  if (x.deviceId() != device_id) {
    throw std::runtime_error(
        "cudaRmsNormDeviceInput: input tensor is on a different CUDA device");
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error(
        "cudaRmsNormDeviceInput: eps must be finite and positive");
  }

  cudaSetDeviceId(device_id);
  CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
  w_dev.upload(weight.data.data(), weight.size() * sizeof(float));
  return cudaRmsNormDeviceWeight(x, w_dev.data(), weight.shape, eps, device_id);
}

CudaTensor cudaSiluDeviceInput(const CudaTensor& x, int device_id) {
  if (x.deviceId() != device_id) {
    throw std::runtime_error(
        "cudaSiluDeviceInput: input tensor is on a different CUDA device");
  }
  cudaSetDeviceId(device_id);
  CudaTensor y(x.shape(), device_id);

  int n = static_cast<int>(x.size());
  siluKernel<<<gridFor(n), kBlockSize>>>(static_cast<const float*>(x.data()),
                                         static_cast<float*>(y.data()), n);
  checkLastKernel("siluKernel");

  return y;
}

CudaTensor cudaElementwiseMulDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b,
                                         int device_id) {
  requireSameShape(a, b, "cudaElementwiseMulDeviceInput");
  if (a.deviceId() != device_id) {
    throw std::runtime_error(
        "cudaElementwiseMulDeviceInput: input tensor is on a different CUDA "
        "device");
  }
  cudaSetDeviceId(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  elementwiseMulKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b.data()),
      static_cast<float*>(y.data()), n);
  checkLastKernel("elementwiseMulKernel");

  return y;
}

CudaTensor cudaElementwiseAddDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b,
                                         int device_id) {
  requireSameShape(a, b, "cudaElementwiseAddDeviceInput");
  if (a.deviceId() != device_id) {
    throw std::runtime_error(
        "cudaElementwiseAddDeviceInput: input tensor is on a different CUDA "
        "device");
  }
  cudaSetDeviceId(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  elementwiseAddKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b.data()),
      static_cast<float*>(y.data()), n);
  checkLastKernel("elementwiseAddKernel");

  return y;
}

CudaTensor cudaElementwiseAddDeviceWeight(const CudaTensor& a,
                                          const void* b_data,
                                          const std::vector<int>& b_shape,
                                          int device_id) {
  validateAddDeviceWeight(a, b_data, b_shape, device_id);
  cudaSetDeviceId(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  elementwiseAddKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b_data),
      static_cast<float*>(y.data()), n);
  checkLastKernel("elementwiseAddKernel");

  return y;
}

// ---------------------------------------------------------------------------
// Embedding Lookup: y[i] = embedding[token_id * dim + i]
// ---------------------------------------------------------------------------
__global__ void embeddingLookupKernel(const float* embedding, float* y,
                                      int token_id, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) {
    y[i] = embedding[token_id * dim + i];
  }
}

// ---------------------------------------------------------------------------
// Softmax Step 1: find max value (for numerical stability)
//
// Uses shared memory tree reduction to find the maximum.
// Launched with <<<1, kBlockSize>>> — one block handles the whole array.
// ---------------------------------------------------------------------------
__global__ void softmaxMaxKernel(const float* x, float* max_out, int n) {
  __shared__ float shared[kBlockSize];
  int tid = threadIdx.x;

  // Each thread finds local max over its stride of elements.
  float local_max = -3.402823466e+38F;  // -FLT_MAX
  for (int i = tid; i < n; i += blockDim.x) {
    local_max = fmaxf(local_max, x[i]);
  }
  shared[tid] = local_max;
  __syncthreads();

  // Tree reduction: threads 0..stride-1 combine pairs.
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }

  // Thread 0 writes the global max.
  if (tid == 0) {
    *max_out = shared[0];
  }
}

// ---------------------------------------------------------------------------
// Softmax Step 2: compute exp(x[i] - max) and their sum
//
// Also uses shared memory tree reduction for the sum.
// Writes exp values to y[] and the sum to sum_out.
// ---------------------------------------------------------------------------
__global__ void softmaxExpSumKernel(const float* x, float* y,
                                    const float* max_value, float* sum_out,
                                    int n) {
  __shared__ float shared[kBlockSize];
  int tid = threadIdx.x;
  float local_sum = 0.0f;
  float max_v = *max_value;

  // Each thread computes exp and accumulates local sum.
  for (int i = tid; i < n; i += blockDim.x) {
    float e = expf(x[i] - max_v);
    y[i] = e;
    local_sum += e;
  }
  shared[tid] = local_sum;
  __syncthreads();

  // Tree reduction for sum.
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *sum_out = shared[0];
  }
}

// ---------------------------------------------------------------------------
// Softmax Step 3: normalize by sum
// ---------------------------------------------------------------------------
__global__ void softmaxNormKernel(float* y, const float* sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] /= *sum;
  }
}

// ---------------------------------------------------------------------------
// RoPE Normal style: pairs are (x[2d], x[2d+1]) within each head
//
// For each pair (x0, x1) at dimension d:
//   x0' = x0 * cos(pos * freq_d) - x1 * sin(pos * freq_d)
//   x1' = x0 * sin(pos * freq_d) + x1 * cos(pos * freq_d)
//   where freq_d = 1 / theta^(2d / head_dim)
// ---------------------------------------------------------------------------
__global__ void ropeNormalKernel(float* x, int n_heads, int head_dim, int pos,
                                 float theta) {
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs_per_head = head_dim / 2;
  int total_pairs = n_heads * pairs_per_head;
  if (pair_index >= total_pairs) {
    return;
  }

  int head = pair_index / pairs_per_head;
  int pair = pair_index % pairs_per_head;
  int dim = pair * 2;
  int base = head * head_dim + dim;

  float freq = 1.0f / powf(theta, static_cast<float>(dim) /
                                      static_cast<float>(head_dim));
  float cos_val = cosf(static_cast<float>(pos) * freq);
  float sin_val = sinf(static_cast<float>(pos) * freq);
  float x0 = x[base];
  float x1 = x[base + 1];
  x[base] = x0 * cos_val - x1 * sin_val;
  x[base + 1] = x0 * sin_val + x1 * cos_val;
}

// ---------------------------------------------------------------------------
// RoPE NeoX style (used by LLaMA): pairs are (x[d], x[d + half_dim])
//
// The first half and second half of each head are paired cross-wise.
// This is the format LLaMA models use in GGUF.
// ---------------------------------------------------------------------------
__global__ void ropeNeoXKernel(float* x, int n_heads, int head_dim, int pos,
                               float theta) {
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  int half_dim = head_dim / 2;
  int total_pairs = n_heads * half_dim;
  if (pair_index >= total_pairs) {
    return;
  }

  int head = pair_index / half_dim;
  int pair = pair_index % half_dim;
  int base = head * head_dim;

  float freq = 1.0f / powf(theta, static_cast<float>(2 * pair) /
                                      static_cast<float>(head_dim));
  float cos_val = cosf(static_cast<float>(pos) * freq);
  float sin_val = sinf(static_cast<float>(pos) * freq);
  float x0 = x[base + pair];
  float x1 = x[base + half_dim + pair];
  x[base + pair] = x0 * cos_val - x1 * sin_val;
  x[base + half_dim + pair] = x0 * sin_val + x1 * cos_val;
}


// ---------------------------------------------------------------------------
// Softmax (host interface)
// ---------------------------------------------------------------------------

Tensor cudaSoftmax(const Tensor& x, int device_id) {
  require1D(x, "cudaSoftmax");
  cudaSetDeviceId(device_id);
  Tensor y(x.shape, 0.0f);

  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  CudaDeviceBuffer max_dev(sizeof(float), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);
  x_dev.upload(x.data.data(), x.size() * sizeof(float));

  int n = static_cast<int>(x.size());

  // Step 1: find max
  softmaxMaxKernel<<<1, kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<float*>(max_dev.data()), n);
  checkLastKernel("softmaxMaxKernel");

  // Step 2: exp(x - max) and sum
  softmaxExpSumKernel<<<1, kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<float*>(y_dev.data()),
      static_cast<const float*>(max_dev.data()),
      static_cast<float*>(sum_dev.data()), n);
  checkLastKernel("softmaxExpSumKernel");

  // Step 3: normalize
  softmaxNormKernel<<<gridFor(n), kBlockSize>>>(
      static_cast<float*>(y_dev.data()),
      static_cast<const float*>(sum_dev.data()), n);
  checkLastKernel("softmaxNormKernel");

  y_dev.download(y.data.data(), y.size() * sizeof(float));
  return y;
}

// ---------------------------------------------------------------------------
// Embedding Lookup (device weight interface)
// ---------------------------------------------------------------------------

CudaTensor cudaEmbeddingLookupDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    int token_id, int device_id) {
  if (embedding_data == nullptr) {
    throw std::runtime_error(
        "cudaEmbeddingLookupDeviceWeight: embedding data is null");
  }
  if (embedding_shape.size() != 2) {
    throw std::runtime_error(
        "cudaEmbeddingLookupDeviceWeight: expected 2D embedding weight");
  }
  int vocab_size = embedding_shape[0];
  int dim = embedding_shape[1];
  if (vocab_size <= 0 || dim <= 0) {
    throw std::runtime_error(
        "cudaEmbeddingLookupDeviceWeight: embedding shape must be positive");
  }
  if (token_id < 0 || token_id >= vocab_size) {
    throw std::out_of_range(
        "cudaEmbeddingLookupDeviceWeight: token id out of range");
  }

  cudaSetDeviceId(device_id);
  CudaTensor y({dim}, device_id);
  embeddingLookupKernel<<<gridFor(dim), kBlockSize>>>(
      static_cast<const float*>(embedding_data),
      static_cast<float*>(y.data()), token_id, dim);
  checkLastKernel("embeddingLookupKernel");
  return y;
}

// ---------------------------------------------------------------------------
// RoPE (host interface, in-place)
// ---------------------------------------------------------------------------

void cudaRope(Tensor& q, Tensor& k, int pos, float theta,
              RopeType rope_type, int device_id) {
  if (q.numDims() != 2 || k.numDims() != 2) {
    throw std::runtime_error("cudaRope: expected 2D tensors");
  }
  if (pos < 0) {
    throw std::out_of_range("cudaRope: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error("cudaRope: theta must be finite and positive");
  }
  if (q.shape[1] != k.shape[1]) {
    throw std::runtime_error("cudaRope: q and k head_dim mismatch");
  }
  if (q.shape[1] <= 0 || q.shape[1] % 2 != 0) {
    throw std::runtime_error("cudaRope: head_dim must be positive and even");
  }

  cudaSetDeviceId(device_id);
  CudaDeviceBuffer q_dev(q.size() * sizeof(float), device_id);
  CudaDeviceBuffer k_dev(k.size() * sizeof(float), device_id);
  q_dev.upload(q.data.data(), q.size() * sizeof(float));
  k_dev.upload(k.data.data(), k.size() * sizeof(float));

  int q_pairs = q.shape[0] * (q.shape[1] / 2);
  int k_pairs = k.shape[0] * (k.shape[1] / 2);

  if (rope_type == RopeType::kNeoX) {
    ropeNeoXKernel<<<gridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q_dev.data()), q.shape[0], q.shape[1], pos, theta);
    checkLastKernel("ropeNeoXKernel(q)");
    ropeNeoXKernel<<<gridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k_dev.data()), k.shape[0], k.shape[1], pos, theta);
    checkLastKernel("ropeNeoXKernel(k)");
  } else {
    ropeNormalKernel<<<gridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q_dev.data()), q.shape[0], q.shape[1], pos, theta);
    checkLastKernel("ropeNormalKernel(q)");
    ropeNormalKernel<<<gridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k_dev.data()), k.shape[0], k.shape[1], pos, theta);
    checkLastKernel("ropeNormalKernel(k)");
  }

  q_dev.download(q.data.data(), q.size() * sizeof(float));
  k_dev.download(k.data.data(), k.size() * sizeof(float));
}

// ---------------------------------------------------------------------------
// RoPE (device interface, in-place)
// ---------------------------------------------------------------------------

void cudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                         int n_kv_heads, int head_dim, int pos, float theta,
                         RopeType rope_type, int device_id) {
  if (q.deviceId() != device_id || k.deviceId() != device_id) {
    throw std::runtime_error(
        "cudaRopeDeviceInput: input tensor on different CUDA device");
  }
  if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
    throw std::runtime_error("cudaRopeDeviceInput: invalid head shape");
  }
  if (q.size() != static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
    throw std::runtime_error("cudaRopeDeviceInput: q shape mismatch");
  }
  if (k.size() !=
      static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim)) {
    throw std::runtime_error("cudaRopeDeviceInput: k shape mismatch");
  }
  if (pos < 0) {
    throw std::out_of_range("cudaRopeDeviceInput: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error("cudaRopeDeviceInput: theta must be finite and positive");
  }

  cudaSetDeviceId(device_id);
  int q_pairs = n_heads * (head_dim / 2);
  int k_pairs = n_kv_heads * (head_dim / 2);

  if (rope_type == RopeType::kNeoX) {
    ropeNeoXKernel<<<gridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q.data()), n_heads, head_dim, pos, theta);
    checkLastKernel("ropeNeoXKernel(q)");
    ropeNeoXKernel<<<gridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k.data()), n_kv_heads, head_dim, pos, theta);
    checkLastKernel("ropeNeoXKernel(k)");
  } else {
    ropeNormalKernel<<<gridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q.data()), n_heads, head_dim, pos, theta);
    checkLastKernel("ropeNormalKernel(q)");
    ropeNormalKernel<<<gridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k.data()), n_kv_heads, head_dim, pos, theta);
    checkLastKernel("ropeNormalKernel(k)");
  }
}

}  // namespace mini_llama