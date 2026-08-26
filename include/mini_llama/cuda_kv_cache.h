#ifndef MINI_LLAMA_CUDA_KV_CACHE_H_
#define MINI_LLAMA_CUDA_KV_CACHE_H_

#include <cstddef>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool cudaKvCacheBuilt();

// GPU-resident KV cache for autoregressive Transformer decoding.
//
// Memory layout (contiguous, for both keys and values):
//   [n_layers][max_seq_len][n_kv_heads][head_dim]
//
// Each "slot" at (layer, pos) holds n_kv_heads * head_dim floats,
// which is the concatenated K or V vectors for all KV heads at that
// layer and position.
//
// Usage:
//   CudaKvCache cache(n_layers, max_seq, n_kv_heads, head_dim, device);
//   cache.write(layer, pos, k, v);          // host -> device
//   cache.writeDevice(layer, pos, k, v);    // device -> device (zero-copy)
//   const void* keys = cache.keysData();    // raw ptr for attention kernel
class CudaKvCache {
public:
    CudaKvCache() = default;
    CudaKvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
                int device_id = 0);

    CudaKvCache(const CudaKvCache&) = delete;
    CudaKvCache& operator=(const CudaKvCache&) = delete;
    CudaKvCache(CudaKvCache&&) noexcept = default;
    CudaKvCache& operator=(CudaKvCache&&) noexcept = default;

    // Releases all device memory and resets parameters.
    void reset();
    void reset(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
               int device_id = 0);

    // Fills both caches with zeros.
    void clear();

    // Writes K/V for one (layer, pos) from host tensors.
    // k, v: [n_kv_heads, head_dim]
    void write(int layer, int pos, const Tensor& k, const Tensor& v);

    // Writes K/V for one (layer, pos) from device tensors (DeviceToDevice).
    // k, v: CudaTensor of shape [n_kv_heads, head_dim]
    void writeDevice(int layer, int pos, const CudaTensor& k,
                     const CudaTensor& v);

    // Reads all heads' K or V for one (layer, pos) back to host.
    Tensor readKey(int layer, int pos) const;
    Tensor readValue(int layer, int pos) const;

    // Reads a single head's K or V for one (layer, pos, kv_head).
    Tensor readKeyHead(int layer, int pos, int kv_head) const;
    Tensor readValueHead(int layer, int pos, int kv_head) const;

    // Raw device pointers for passing to attention kernels.
    const void* keysData() const;
    const void* valuesData() const;

    bool empty() const;
    size_t bytes() const;

    int nLayers() const { return n_layers_; }
    int maxSeqLen() const { return max_seq_len_; }
    int nKvHeads() const { return n_kv_heads_; }
    int headDim() const { return head_dim_; }
    int deviceId() const { return device_id_; }

private:
    void validateIndices(int layer, int pos) const;
    void validateHeadIndex(int kv_head) const;
    void validateWriteTensors(const Tensor& k, const Tensor& v) const;

    // Byte offset into the flat buffer for a given (layer, pos) slot.
    size_t slotOffsetBytes(int layer, int pos) const;

    // Byte offset for a specific (layer, pos, kv_head) within the buffer.
    size_t headOffsetBytes(int layer, int pos, int kv_head) const;

    int n_layers_ = 0;
    int max_seq_len_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int device_id_ = 0;
    CudaDeviceBuffer keys_;
    CudaDeviceBuffer values_;
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_CUDA_KV_CACHE_H_