#include "mini_llama/cuda_attention.h"

#include <stdexcept>

namespace mini_llama {
namespace {

    std::runtime_error cudaAttentionNotBuiltError() {
        return std::runtime_error(
            "CUDA attention was not built. Reconfigure with "
            "-DMINI_LLAMA_CUDA=ON on "
            "a NVIDIA CUDA machine.");
    }

}  // namespace

bool cudaAttentionBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA

Tensor cudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache,
                           int layer, int pos, int n_heads, int n_kv_heads,
                           int head_dim, int device_id) {
    (void)q;
    (void)kv_cache;
    (void)layer;
    (void)pos;
    (void)n_heads;
    (void)n_kv_heads;
    (void)head_dim;
    (void)device_id;
    throw cudaAttentionNotBuiltError();
}

CudaTensor cudaAttentionDecodeDeviceInput(const CudaTensor& q,
                                          const CudaKvCache& kv_cache,
                                          int layer, int pos, int n_heads,
                                          int n_kv_heads, int head_dim,
                                          int device_id) {
    (void)q;
    (void)kv_cache;
    (void)layer;
    (void)pos;
    (void)n_heads;
    (void)n_kv_heads;
    (void)head_dim;
    (void)device_id;
    throw cudaAttentionNotBuiltError();
}

#endif  // MINI_LLAMA_USE_CUDA

}  // namespace mini_llama