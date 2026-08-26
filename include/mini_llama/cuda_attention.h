#ifndef MINI_LLAMA_CUDA_ATTENTION_H_
#define MINI_LLAMA_CUDA_ATTENTION_H_

#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool cudaAttentionBuilt();

// Decode-phase attention with host input.
// q: [n_heads, head_dim] on CPU
// Returns: [n_heads, head_dim] on CPU
Tensor cudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache,
                           int layer, int pos, int n_heads, int n_kv_heads,
                           int head_dim, int device_id = 0);

// Decode-phase attention with device input (zero-copy pipeline).
// q: CudaTensor [n_heads * head_dim] on GPU
// Returns: CudaTensor [n_heads * head_dim] on GPU
CudaTensor cudaAttentionDecodeDeviceInput(const CudaTensor& q,
                                          const CudaKvCache& kv_cache,
                                          int layer, int pos, int n_heads,
                                          int n_kv_heads, int head_dim,
                                          int device_id = 0);

}  // namespace mini_llama

#endif  // MINI_LLAMA_CUDA_ATTENTION_H_