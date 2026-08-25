#include "mini_llama/cuda_kv_cache.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {
namespace {

    std::runtime_error cudaKvCacheNotBuiltError() {
        return std::runtime_error(
            "CUDA KV cache was not built. Reconfigure with "
            "-DMINI_LLAMA_CUDA=ON on "
            "a NVIDIA CUDA machine.");
    }

    void requirePositive(int value, const char* name) {
        if (value <= 0) {
            throw std::runtime_error(std::string("CudaKvCache: ") + name +
                                     " must be positive");
        }
    }

    // Overflow-checked multiplication.
    size_t checkedMul(size_t a, size_t b, const char* label) {
        if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
            throw std::runtime_error(
                std::string("CudaKvCache: size overflow for ") + label);
        }
        return a * b;
    }

}  // namespace

// ===========================================================================
// Public API
// ===========================================================================

bool cudaKvCacheBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

CudaKvCache::CudaKvCache(int n_layers, int max_seq_len, int n_kv_heads,
                         int head_dim, int device_id) {
    reset(n_layers, max_seq_len, n_kv_heads, head_dim, device_id);
}

void CudaKvCache::reset() {
    keys_.reset();
    values_.reset();
    n_layers_ = 0;
    max_seq_len_ = 0;
    n_kv_heads_ = 0;
    head_dim_ = 0;
    device_id_ = 0;
}

void CudaKvCache::reset(int n_layers, int max_seq_len, int n_kv_heads,
                        int head_dim, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    requirePositive(n_layers, "n_layers");
    requirePositive(max_seq_len, "max_seq_len");
    requirePositive(n_kv_heads, "n_kv_heads");
    requirePositive(head_dim, "head_dim");

    // Total elements = n_layers * max_seq_len * n_kv_heads * head_dim
    // Each of keys and values needs this many floats.
    size_t elements = static_cast<size_t>(n_layers);
    elements =
        checkedMul(elements, static_cast<size_t>(max_seq_len), "layers * seq");
    elements = checkedMul(elements, static_cast<size_t>(n_kv_heads),
                          "layers * seq * heads");
    elements = checkedMul(elements, static_cast<size_t>(head_dim),
                          "layers * seq * heads * dim");
    size_t bytes_per_cache = checkedMul(elements, sizeof(float), "cache bytes");

    n_layers_ = n_layers;
    max_seq_len_ = max_seq_len;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    device_id_ = device_id;
    keys_.reset(bytes_per_cache, device_id);
    values_.reset(bytes_per_cache, device_id);
    clear();
#else
    (void)n_layers;
    (void)max_seq_len;
    (void)n_kv_heads;
    (void)head_dim;
    (void)device_id;
    throw cudaKvCacheNotBuiltError();
#endif
}

void CudaKvCache::clear() {
#ifdef MINI_LLAMA_USE_CUDA
    if (empty()) {
        return;
    }
    cudaSetDeviceId(device_id_);
    // Zero-fill by uploading a zero vector.
    // Alternative: cudaMemset for large caches (avoids host allocation).
    std::vector<float> zeros(keys_.bytes() / sizeof(float), 0.0f);
    keys_.upload(zeros.data(), keys_.bytes());
    values_.upload(zeros.data(), values_.bytes());
#else
    throw cudaKvCacheNotBuiltError();
#endif
}

// ---------------------------------------------------------------------------
// Write: host -> device
// ---------------------------------------------------------------------------

void CudaKvCache::write(int layer, int pos, const Tensor& k, const Tensor& v) {
#ifdef MINI_LLAMA_USE_CUDA
    validateIndices(layer, pos);
    validateWriteTensors(k, v);

    size_t offset = slotOffsetBytes(layer, pos);
    size_t bytes_to_copy = k.size() * sizeof(float);

    // Pointer arithmetic: offset into the flat device buffer.
    char* key_dst = static_cast<char*>(keys_.data()) + offset;
    char* value_dst = static_cast<char*>(values_.data()) + offset;

    cudaSetDeviceId(device_id_);
    cudaMemcpyBytes(key_dst, k.data.data(), bytes_to_copy,
                    CudaMemcpyKind::kHostToDevice);
    cudaMemcpyBytes(value_dst, v.data.data(), bytes_to_copy,
                    CudaMemcpyKind::kHostToDevice);
#else
    (void)layer;
    (void)pos;
    (void)k;
    (void)v;
    throw cudaKvCacheNotBuiltError();
#endif
}

// ---------------------------------------------------------------------------
// WriteDevice: device -> device (zero-copy pipeline)
// ---------------------------------------------------------------------------

void CudaKvCache::writeDevice(int layer, int pos, const CudaTensor& k,
                              const CudaTensor& v) {
#ifdef MINI_LLAMA_USE_CUDA
    validateIndices(layer, pos);
    if (k.deviceId() != device_id_ || v.deviceId() != device_id_) {
        throw std::runtime_error(
            "CudaKvCache::writeDevice tensor is on a different CUDA device");
    }
    const size_t expected =
        static_cast<size_t>(n_kv_heads_) * static_cast<size_t>(head_dim_);
    if (k.size() != expected || v.size() != expected) {
        throw std::out_of_range(
            "CudaKvCache::writeDevice tensor shape does not match cache shape");
    }

    size_t offset = slotOffsetBytes(layer, pos);
    size_t bytes_to_copy = expected * sizeof(float);

    char* key_dst = static_cast<char*>(keys_.data()) + offset;
    char* value_dst = static_cast<char*>(values_.data()) + offset;

    cudaSetDeviceId(device_id_);
    cudaMemcpyBytes(key_dst, k.data(), bytes_to_copy,
                    CudaMemcpyKind::kDeviceToDevice);
    cudaMemcpyBytes(value_dst, v.data(), bytes_to_copy,
                    CudaMemcpyKind::kDeviceToDevice);
#else
    (void)layer;
    (void)pos;
    (void)k;
    (void)v;
    throw cudaKvCacheNotBuiltError();
#endif
}

// ---------------------------------------------------------------------------
// Read: device -> host (for debugging / testing)
// ---------------------------------------------------------------------------

Tensor CudaKvCache::readKey(int layer, int pos) const {
#ifdef MINI_LLAMA_USE_CUDA
    validateIndices(layer, pos);
    Tensor out({n_kv_heads_, head_dim_}, 0.0f);
    const char* src =
        static_cast<const char*>(keys_.data()) + slotOffsetBytes(layer, pos);
    cudaSetDeviceId(device_id_);
    cudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float),
                    CudaMemcpyKind::kDeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    throw cudaKvCacheNotBuiltError();
#endif
}

Tensor CudaKvCache::readValue(int layer, int pos) const {
#ifdef MINI_LLAMA_USE_CUDA
    validateIndices(layer, pos);
    Tensor out({n_kv_heads_, head_dim_}, 0.0f);
    const char* src =
        static_cast<const char*>(values_.data()) + slotOffsetBytes(layer, pos);
    cudaSetDeviceId(device_id_);
    cudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float),
                    CudaMemcpyKind::kDeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    throw cudaKvCacheNotBuiltError();
#endif
}

Tensor CudaKvCache::readKeyHead(int layer, int pos, int kv_head) const {
#ifdef MINI_LLAMA_USE_CUDA
    validateIndices(layer, pos);
    validateHeadIndex(kv_head);
    Tensor out({head_dim_}, 0.0f);
    const char* src = static_cast<const char*>(keys_.data()) +
                      headOffsetBytes(layer, pos, kv_head);
    cudaSetDeviceId(device_id_);
    cudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float),
                    CudaMemcpyKind::kDeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    (void)kv_head;
    throw cudaKvCacheNotBuiltError();
#endif
}

Tensor CudaKvCache::readValueHead(int layer, int pos, int kv_head) const {
#ifdef MINI_LLAMA_USE_CUDA
    validateIndices(layer, pos);
    validateHeadIndex(kv_head);
    Tensor out({head_dim_}, 0.0f);
    const char* src = static_cast<const char*>(values_.data()) +
                      headOffsetBytes(layer, pos, kv_head);
    cudaSetDeviceId(device_id_);
    cudaMemcpyBytes(out.data.data(), src, out.size() * sizeof(float),
                    CudaMemcpyKind::kDeviceToHost);
    return out;
#else
    (void)layer;
    (void)pos;
    (void)kv_head;
    throw cudaKvCacheNotBuiltError();
#endif
}

// ---------------------------------------------------------------------------
// Raw device pointers (for attention kernels)
// ---------------------------------------------------------------------------

const void* CudaKvCache::keysData() const {
    if (empty()) {
        throw std::runtime_error("CudaKvCache: cache is empty");
    }
    return keys_.data();
}

const void* CudaKvCache::valuesData() const {
    if (empty()) {
        throw std::runtime_error("CudaKvCache: cache is empty");
    }
    return values_.data();
}

bool CudaKvCache::empty() const { return keys_.empty() || values_.empty(); }

size_t CudaKvCache::bytes() const { return keys_.bytes() + values_.bytes(); }

// ---------------------------------------------------------------------------
// Private: validation and offset computation
// ---------------------------------------------------------------------------

void CudaKvCache::validateIndices(int layer, int pos) const {
    if (empty()) {
        throw std::runtime_error("CudaKvCache: cache is empty");
    }
    if (layer < 0 || layer >= n_layers_) {
        throw std::out_of_range("CudaKvCache layer out of range");
    }
    if (pos < 0 || pos >= max_seq_len_) {
        throw std::out_of_range("CudaKvCache position out of range");
    }
}

void CudaKvCache::validateHeadIndex(int kv_head) const {
    if (kv_head < 0 || kv_head >= n_kv_heads_) {
        throw std::out_of_range("CudaKvCache head out of range");
    }
}

void CudaKvCache::validateWriteTensors(const Tensor& k, const Tensor& v) const {
    if (k.numDims() != 2 || v.numDims() != 2 || k.shape != v.shape) {
        throw std::out_of_range(
            "CudaKvCache::write expected matching [n_kv_heads, head_dim] "
            "tensors");
    }
    if (k.shape[0] != n_kv_heads_ || k.shape[1] != head_dim_) {
        throw std::out_of_range(
            "CudaKvCache::write tensor shape does not match cache shape");
    }
}

// Byte offset for the start of a (layer, pos) slot.
// Layout: [n_layers][max_seq_len][n_kv_heads][head_dim]
// Slot index = (layer * max_seq_len + pos) * n_kv_heads * head_dim
size_t CudaKvCache::slotOffsetBytes(int layer, int pos) const {
    size_t slot = static_cast<size_t>(layer);
    slot = slot * static_cast<size_t>(max_seq_len_) + static_cast<size_t>(pos);
    slot = slot * static_cast<size_t>(n_kv_heads_) *
           static_cast<size_t>(head_dim_);
    return slot * sizeof(float);
}

// Byte offset for a specific head within a (layer, pos) slot.
size_t CudaKvCache::headOffsetBytes(int layer, int pos, int kv_head) const {
    size_t offset = slotOffsetBytes(layer, pos);
    offset += static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim_) *
              sizeof(float);
    return offset;
}

}  // namespace mini_llama