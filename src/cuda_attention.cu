
#include "mini_llama/cuda_attention.h"

#ifdef MINI_LLAMA_USE_CUDA

#include <cuda_runtime_api.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"

namespace mini_llama {
namespace {

constexpr int kAttnBlockSize = 128;

void checkCudaAttention(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA attention error in " + std::string(expr) +
                             ": " + cudaGetErrorString(err));
  }
}

void checkLastAttentionKernel(const char* name) {
  checkCudaAttention(cudaGetLastError(), name);
  checkCudaAttention(cudaDeviceSynchronize(), name);
}

size_t checkedMul(size_t a, size_t b, const char* label) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    throw std::runtime_error(
        std::string("cudaAttentionDecode: size overflow for ") + label);
  }
  return a * b;
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

void validateAttentionInputs(const Tensor& q, const CudaKvCache& kv_cache,
                             int layer, int pos, int n_heads,
                             int n_kv_heads, int head_dim) {
  if (q.shape != std::vector<int>{n_heads, head_dim}) {
    throw std::runtime_error(
        "cudaAttentionDecode: expected q shape [" +
        std::to_string(n_heads) + ", " + std::to_string(head_dim) +
        "], got " + q.shapeString());
  }
  if (kv_cache.empty()) {
    throw std::runtime_error("cudaAttentionDecode: KV cache is empty");
  }
  if (layer < 0 || layer >= kv_cache.nLayers()) {
    throw std::out_of_range("cudaAttentionDecode: layer out of range");
  }
  if (pos < 0 || pos >= kv_cache.maxSeqLen()) {
    throw std::out_of_range("cudaAttentionDecode: position out of range");
  }
  if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
    throw std::runtime_error(
        "cudaAttentionDecode: head counts and head_dim must be positive");
  }
  if (n_heads % n_kv_heads != 0) {
    throw std::runtime_error(
        "cudaAttentionDecode: n_heads must be divisible by n_kv_heads");
  }
  if (kv_cache.nKvHeads() != n_kv_heads ||
      kv_cache.headDim() != head_dim) {
    throw std::runtime_error(
        "cudaAttentionDecode: KV cache shape does not match attention shape");
  }
}

void validateAttentionDeviceInputs(const CudaTensor& q,
                                   const CudaKvCache& kv_cache, int layer,
                                   int pos, int n_heads, int n_kv_heads,
                                   int head_dim, int device_id) {
  if (q.deviceId() != device_id) {
    throw std::runtime_error(
        "cudaAttentionDecodeDeviceInput: q on different CUDA device");
  }
  if (q.size() !=
      static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
    throw std::runtime_error(
        "cudaAttentionDecodeDeviceInput: q shape mismatch");
  }
  if (kv_cache.empty()) {
    throw std::runtime_error(
        "cudaAttentionDecodeDeviceInput: KV cache is empty");
  }
  if (layer < 0 || layer >= kv_cache.nLayers()) {
    throw std::out_of_range(
        "cudaAttentionDecodeDeviceInput: layer out of range");
  }
  if (pos < 0 || pos >= kv_cache.maxSeqLen()) {
    throw std::out_of_range(
        "cudaAttentionDecodeDeviceInput: position out of range");
  }
  if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
    throw std::runtime_error(
        "cudaAttentionDecodeDeviceInput: head counts and head_dim must be "
        "positive");
  }
  if (n_heads % n_kv_heads != 0) {
    throw std::runtime_error(
        "cudaAttentionDecodeDeviceInput: n_heads must be divisible by "
        "n_kv_heads");
  }
  if (kv_cache.nKvHeads() != n_kv_heads ||
      kv_cache.headDim() != head_dim ||
      kv_cache.deviceId() != device_id) {
    throw std::runtime_error(
        "cudaAttentionDecodeDeviceInput: KV cache shape or device mismatch");
  }
}

// ===========================================================================
// CUDA Kernels
// ===========================================================================
//
// Decode-phase attention: one query token against all cached K/V.
//
// Two-step process:
//   1. AttentionScoresKernel: compute Q·K^T for all (head, position) pairs
//   2. AttentionReduceKernel: online softmax + weighted sum of V

// ---------------------------------------------------------------------------
// Step 1: Attention Scores = Q · K^T
//
// Grid: blockIdx.x = head, blockIdx.y = cached position t
// Each block's threads cooperatively compute the dot product
// Q[head, :] · K[t, kv_head, :] using shared memory reduction.
//
// GQA: multiple query heads share one KV head.
//   kv_group = n_heads / n_kv_heads
//   kv_head = head / kv_group
// ---------------------------------------------------------------------------
__global__ void attentionScoresKernel(
    const float* q, const float* keys, float* scores,
    int layer, int max_seq_len, int n_heads, int n_kv_heads,
    int head_dim, int pos, float scale) {
  int h = blockIdx.x;   // query head index
  int t = blockIdx.y;   // cached position index
  if (h >= n_heads || t > pos) {
    return;
  }

  // GQA: map query head to KV head
  int kv_group = n_heads / n_kv_heads;
  int kv_head = h / kv_group;

  // Index into flat KV cache: [layer][max_seq][n_kv_heads][head_dim]
  size_t key_base =
      ((static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) +
        static_cast<size_t>(t)) *
           static_cast<size_t>(n_kv_heads) +
       static_cast<size_t>(kv_head)) *
      static_cast<size_t>(head_dim);
  size_t q_base = static_cast<size_t>(h) * static_cast<size_t>(head_dim);

  // Cooperative dot product with shared memory reduction
  __shared__ float partial[kAttnBlockSize];
  float sum = 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    sum += q[q_base + d] * keys[key_base + d];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scores[static_cast<size_t>(h) * static_cast<size_t>(pos + 1) +
           static_cast<size_t>(t)] = partial[0] * scale;
  }
}

// ---------------------------------------------------------------------------
// Step 2: Online Softmax + Weighted Sum of V (fused kernel)
//
// Grid: blockIdx.x = head
// Each block handles one head's full softmax + V reduction.
//
// For each head h:
//   1. Find max score (shared memory reduction)
//   2. Compute sum of exp(score - max) (shared memory reduction)
//   3. For each output dimension d:
//        out[h, d] = sum_t( softmax(score[h,t]) * V[t, kv_head, d] )
//
// This fuses softmax normalization and V multiplication into one kernel,
// avoiding a separate softmax pass and its memory traffic.
// ---------------------------------------------------------------------------
__global__ void attentionReduceKernel(
    const float* scores, const float* values, float* out,
    int layer, int max_seq_len, int n_heads, int n_kv_heads,
    int head_dim, int pos) {
  int h = blockIdx.x;
  if (h >= n_heads) {
    return;
  }

  int kv_group = n_heads / n_kv_heads;
  int kv_head = h / kv_group;
  size_t score_base = static_cast<size_t>(h) * static_cast<size_t>(pos + 1);

  __shared__ float partial[kAttnBlockSize];

  // --- Pass 1: find max score ---
  float local_max = -INFINITY;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    local_max = fmaxf(local_max, scores[score_base + static_cast<size_t>(t)]);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] =
          fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float max_score = partial[0];

  // --- Pass 2: sum of exp(score - max) ---
  float local_sum = 0.0f;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    local_sum += expf(scores[score_base + static_cast<size_t>(t)] - max_score);
  }
  partial[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float denom = partial[0];

  // --- Pass 3: weighted sum of V ---
  // Each thread handles one (or more) output dimension(s).
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; ++t) {
      float prob =
          expf(scores[score_base + static_cast<size_t>(t)] - max_score) /
          denom;
      size_t value_idx =
          ((static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) +
            static_cast<size_t>(t)) *
               static_cast<size_t>(n_kv_heads) +
           static_cast<size_t>(kv_head)) *
              static_cast<size_t>(head_dim) +
          static_cast<size_t>(d);
      value += prob * values[value_idx];
    }
    out[static_cast<size_t>(h) * static_cast<size_t>(head_dim) +
        static_cast<size_t>(d)] = value;
  }
}

}  // namespace

// ===========================================================================
// Public API
// ===========================================================================
Tensor cudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache,
                           int layer, int pos, int n_heads, int n_kv_heads,
                           int head_dim, int device_id) {
  validateAttentionInputs(q, kv_cache, layer, pos, n_heads, n_kv_heads,
                          head_dim);
  cudaSetDeviceId(device_id);

  size_t q_bytes = checkedMul(q.size(), sizeof(float), "q bytes");
  size_t out_elements =
      checkedMul(static_cast<size_t>(n_heads), static_cast<size_t>(head_dim),
                 "output elements");
  size_t out_bytes = checkedMul(out_elements, sizeof(float), "output bytes");
  size_t score_elements =
      checkedMul(static_cast<size_t>(n_heads), static_cast<size_t>(pos + 1),
                 "score elements");
  size_t score_bytes =
      checkedMul(score_elements, sizeof(float), "score bytes");

  CudaDeviceBuffer q_dev(q_bytes, device_id);
  CudaDeviceBuffer scores_dev(score_bytes, device_id);
  CudaDeviceBuffer out_dev(out_bytes, device_id);
  q_dev.upload(q.data.data(), q_bytes);

  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  dim3 score_grid(static_cast<unsigned int>(n_heads),
                  static_cast<unsigned int>(pos + 1));
  attentionScoresKernel<<<score_grid, kAttnBlockSize>>>(
      static_cast<const float*>(q_dev.data()),
      static_cast<const float*>(kv_cache.keysData()),
      static_cast<float*>(scores_dev.data()), layer,
      kv_cache.maxSeqLen(), n_heads, n_kv_heads, head_dim, pos, scale);
  checkLastAttentionKernel("attentionScoresKernel");

  attentionReduceKernel<<<n_heads, kAttnBlockSize>>>(
      static_cast<const float*>(scores_dev.data()),
      static_cast<const float*>(kv_cache.valuesData()),
      static_cast<float*>(out_dev.data()), layer,
      kv_cache.maxSeqLen(), n_heads, n_kv_heads, head_dim, pos);
  checkLastAttentionKernel("attentionReduceKernel");

  Tensor out({n_heads, head_dim}, 0.0f);
  out_dev.download(out.data.data(), out_bytes);
  return out;
}

CudaTensor cudaAttentionDecodeDeviceInput(
    const CudaTensor& q, const CudaKvCache& kv_cache,
    int layer, int pos, int n_heads, int n_kv_heads, int head_dim,
    int device_id) {
  validateAttentionDeviceInputs(q, kv_cache, layer, pos, n_heads, n_kv_heads,
                                head_dim, device_id);
  cudaSetDeviceId(device_id);

  size_t out_elements =
      checkedMul(static_cast<size_t>(n_heads), static_cast<size_t>(head_dim),
                 "output elements");
  size_t score_elements =
      checkedMul(static_cast<size_t>(n_heads), static_cast<size_t>(pos + 1),
                 "score elements");
  size_t score_bytes =
      checkedMul(score_elements, sizeof(float), "score bytes");

  CudaDeviceBuffer scores_dev(score_bytes, device_id);
  CudaTensor y({static_cast<int>(out_elements)}, device_id);

  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  dim3 score_grid(static_cast<unsigned int>(n_heads),
                  static_cast<unsigned int>(pos + 1));
  attentionScoresKernel<<<score_grid, kAttnBlockSize>>>(
      static_cast<const float*>(q.data()),
      static_cast<const float*>(kv_cache.keysData()),
      static_cast<float*>(scores_dev.data()), layer,
      kv_cache.maxSeqLen(), n_heads, n_kv_heads, head_dim, pos, scale);
  checkLastAttentionKernel("attentionScoresKernel");

  attentionReduceKernel<<<n_heads, kAttnBlockSize>>>(
      static_cast<const float*>(scores_dev.data()),
      static_cast<const float*>(kv_cache.valuesData()),
      static_cast<float*>(y.data()), layer,
      kv_cache.maxSeqLen(), n_heads, n_kv_heads, head_dim, pos);
  checkLastAttentionKernel("attentionReduceKernel");

  return y;
}

}  // namespace mini_llama

#endif  // MINI_LLAMA_USE_CUDA