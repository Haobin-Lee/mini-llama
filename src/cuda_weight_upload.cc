#include "mini_llama/cuda_weight_upload.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {
namespace {

    void throwIfEmpty(const std::string& name, const std::string& format) {
        throw std::runtime_error("CUDA weight upload: empty " + format +
                                 " data for " + name);
    }

}  // namespace

// ===========================================================================
// F32 linear weight upload
// ===========================================================================

void uploadF32LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    if (weight.type != QuantType::kF32) {
        return;
    }
    if (weight.f32_data.empty()) {
        throwIfEmpty(name, "F32");
    }

    const size_t expected_numel = weight.num_elements();
    if (weight.f32_data.size() != expected_numel) {
        throw std::runtime_error("CUDA weight upload: data size mismatch for " +
                                 name + ", expected " +
                                 std::to_string(expected_numel) + ", got " +
                                 std::to_string(weight.f32_data.size()));
    }

    const size_t bytes = weight.f32_data.size() * sizeof(float);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kF32;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = 0;
    uploaded.buffer.reset(bytes, device_id);
    uploaded.buffer.upload(weight.f32_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
#else
    (void)storage;
    (void)weight;
    (void)name;
    (void)device_id;
    throw std::runtime_error(
        "CUDA weight upload requires CUDA. Reconfigure with "
        "-DMINI_LLAMA_CUDA=ON.");
#endif
}

// ===========================================================================
// F32 tensor weight upload (norm, embedding, bias)
// ===========================================================================

void uploadF32TensorWeight(CudaModelWeights& storage, const Tensor& weight,
                           const std::string& name, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    if (weight.data.empty()) {
        return;
    }

    const size_t expected_numel = weight.size();
    if (weight.data.size() != expected_numel) {
        throw std::runtime_error(
            "CUDA weight upload: tensor data size mismatch for " + name +
            ", expected " + std::to_string(expected_numel) + ", got " +
            std::to_string(weight.data.size()));
    }

    const size_t bytes = weight.data.size() * sizeof(float);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kF32;
    uploaded.linear = false;
    uploaded.shape = weight.shape;
    uploaded.block_count = 0;
    uploaded.buffer.reset(bytes, device_id);
    uploaded.buffer.upload(weight.data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
#else
    (void)storage;
    (void)weight;
    (void)name;
    (void)device_id;
    throw std::runtime_error(
        "CUDA weight upload requires CUDA. Reconfigure with "
        "-DMINI_LLAMA_CUDA=ON.");
#endif
}

// ===========================================================================
// Q8_0 linear weight upload
// ===========================================================================

void uploadQ80LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    if (weight.type != QuantType::kQ80) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error(
            "CUDA weight upload: expected 2D Q8_0 weight for " + name);
    }
    if (weight.q8_0_data.empty()) {
        throwIfEmpty(name, "Q8_0");
    }

    // Each row has ceil(in_features / 32) blocks.
    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row =
        (in_features + kQ80BlockSize - 1) / kQ80BlockSize;
    const size_t expected_blocks =
        static_cast<size_t>(out_features) * blocks_per_row;

    if (weight.q8_0_data.size() != expected_blocks) {
        throw std::runtime_error(
            "CUDA weight upload: Q8_0 block count mismatch for " + name +
            ", expected " + std::to_string(expected_blocks) + ", got " +
            std::to_string(weight.q8_0_data.size()));
    }

    const size_t bytes = weight.q8_0_data.size() * sizeof(BlockQ80);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ80;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q8_0_data.size();
    uploaded.buffer.reset(bytes, device_id);
    uploaded.buffer.upload(weight.q8_0_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
#else
    (void)storage;
    (void)weight;
    (void)name;
    (void)device_id;
    throw std::runtime_error(
        "CUDA weight upload requires CUDA. Reconfigure with "
        "-DMINI_LLAMA_CUDA=ON.");
#endif
}

// ===========================================================================
// Q4_0 linear weight upload
// ===========================================================================

void uploadQ40LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    if (weight.type != QuantType::kQ40) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error(
            "CUDA weight upload: expected 2D Q4_0 weight for " + name);
    }
    if (weight.q4_0_data.empty()) {
        throwIfEmpty(name, "Q4_0");
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row =
        (in_features + kQ40BlockSize - 1) / kQ40BlockSize;
    const size_t expected_blocks =
        static_cast<size_t>(out_features) * blocks_per_row;

    if (weight.q4_0_data.size() != expected_blocks) {
        throw std::runtime_error(
            "CUDA weight upload: Q4_0 block count mismatch for " + name +
            ", expected " + std::to_string(expected_blocks) + ", got " +
            std::to_string(weight.q4_0_data.size()));
    }

    const size_t bytes = weight.q4_0_data.size() * sizeof(BlockQ40);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ40;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q4_0_data.size();
    uploaded.buffer.reset(bytes, device_id);
    uploaded.buffer.upload(weight.q4_0_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
#else
    (void)storage;
    (void)weight;
    (void)name;
    (void)device_id;
    throw std::runtime_error(
        "CUDA weight upload requires CUDA. Reconfigure with "
        "-DMINI_LLAMA_CUDA=ON.");
#endif
}

// ===========================================================================
// Q4_1 linear weight upload
// ===========================================================================

void uploadQ41LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    if (weight.type != QuantType::kQ41) {
        return;
    }
    if (weight.shape.size() != 2) {
        throw std::runtime_error(
            "CUDA weight upload: expected 2D Q4_1 weight for " + name);
    }
    if (weight.q4_1_data.empty()) {
        throwIfEmpty(name, "Q4_1");
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row =
        (in_features + kQ41BlockSize - 1) / kQ41BlockSize;
    const size_t expected_blocks =
        static_cast<size_t>(out_features) * blocks_per_row;

    if (weight.q4_1_data.size() != expected_blocks) {
        throw std::runtime_error(
            "CUDA weight upload: Q4_1 block count mismatch for " + name +
            ", expected " + std::to_string(expected_blocks) + ", got " +
            std::to_string(weight.q4_1_data.size()));
    }

    const size_t bytes = weight.q4_1_data.size() * sizeof(BlockQ41);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ41;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q4_1_data.size();
    uploaded.buffer.reset(bytes, device_id);
    uploaded.buffer.upload(weight.q4_1_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
#else
    (void)storage;
    (void)weight;
    (void)name;
    (void)device_id;
    throw std::runtime_error(
        "CUDA weight upload requires CUDA. Reconfigure with "
        "-DMINI_LLAMA_CUDA=ON.");
#endif
}

// ===========================================================================
// Dispatch: upload any linear weight regardless of type
// ===========================================================================

void uploadLinearWeight(CudaModelWeights& storage,
                        const QuantizedTensor& weight, const std::string& name,
                        int device_id) {
    uploadF32LinearWeight(storage, weight, name, device_id);
    uploadQ80LinearWeight(storage, weight, name, device_id);
    uploadQ40LinearWeight(storage, weight, name, device_id);
    uploadQ41LinearWeight(storage, weight, name, device_id);
}

// ===========================================================================
// Runtime stats helpers
// ===========================================================================

void recordCudaLinearCall(CudaModelWeights& storage) {
    storage.linear_calls += 1;
}

void recordCudaLinearCopy(CudaModelWeights& storage, size_t h2d_bytes,
                          size_t d2h_bytes) {
    storage.linear_calls += 1;
    storage.host_to_device_copies += 1;
    storage.device_to_host_copies += 1;
    storage.host_to_device_bytes += h2d_bytes;
    storage.device_to_host_bytes += d2h_bytes;
}

void recordCudaHostToDeviceCopy(CudaModelWeights& storage, size_t bytes) {
    storage.host_to_device_copies += 1;
    storage.host_to_device_bytes += bytes;
}

void resetCudaRuntimeStats(CudaModelWeights& storage) {
    storage.linear_calls = 0;
    storage.activation_calls = 0;
    storage.attention_calls = 0;
    storage.attention_cpu_fallbacks = 0;
    storage.kv_cache_write_bytes = 0;
    storage.kv_cache_read_bytes = 0;
    storage.host_to_device_copies = 0;
    storage.device_to_host_copies = 0;
    storage.host_to_device_bytes = 0;
    storage.device_to_host_bytes = 0;
}

}  // namespace mini_llama
