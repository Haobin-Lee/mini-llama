#ifndef MINI_LLAMA_CUDA_WEIGHT_UPLOAD_H_
#define MINI_LLAMA_CUDA_WEIGHT_UPLOAD_H_

#include <cstddef>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// CudaUploadedWeight: one weight tensor resident on GPU
// ---------------------------------------------------------------------------

struct CudaUploadedWeight {
    // Unique name for lookup, e.g. "layers.0.wq", "token_embedding"
    std::string name;

    // Quantization type of the uploaded data.
    QuantType type = QuantType::kF32;

    // true = linear layer weight (used with matmul/quant kernel)
    // false = norm/embedding/bias (used with elementwise kernel)
    bool linear = true;

    // Shape of the weight in elements.
    // For F32: [out_features, in_features] or [dim]
    // For Q8_0/Q4_0/Q4_1: [out_features, in_features]
    std::vector<int> shape;

    // Number of quantization blocks (0 for F32).
    size_t block_count = 0;

    // Device memory holding the weight data.
    CudaDeviceBuffer buffer;
};

// ---------------------------------------------------------------------------
// CudaModelWeights: all uploaded weights + runtime stats
// ---------------------------------------------------------------------------

struct CudaModelWeights {
    int device_id = 0;

    // Upload stats.
    size_t uploaded_weight_count = 0;
    size_t uploaded_bytes = 0;

    // Runtime stats (incremented during inference).
    size_t linear_calls = 0;
    size_t activation_calls = 0;
    size_t attention_calls = 0;
    size_t attention_cpu_fallbacks = 0;
    size_t kv_cache_write_bytes = 0;
    size_t kv_cache_read_bytes = 0;
    size_t host_to_device_copies = 0;
    size_t device_to_host_copies = 0;
    size_t host_to_device_bytes = 0;
    size_t device_to_host_bytes = 0;

    // All uploaded weights, searched by name + type.
    std::vector<CudaUploadedWeight> weights;
};

// ---------------------------------------------------------------------------
// Upload functions
// ---------------------------------------------------------------------------

// Uploads a F32 linear weight (QuantizedTensor with type==kF32).
// Skips if weight is not F32.
void uploadF32LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id);

// Uploads a F32 tensor weight (norm, embedding, bias).
// Skips if tensor is empty.
void uploadF32TensorWeight(CudaModelWeights& storage, const Tensor& weight,
                           const std::string& name, int device_id);

// Uploads a Q8_0 linear weight. Skips if weight is not Q8_0.
void uploadQ80LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id);

// Uploads a Q4_0 linear weight. Skips if weight is not Q4_0.
void uploadQ40LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id);

// Uploads a Q4_1 linear weight. Skips if weight is not Q4_1.
void uploadQ41LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id);

// Dispatches to the correct upload function based on weight.type.
void uploadLinearWeight(CudaModelWeights& storage,
                        const QuantizedTensor& weight, const std::string& name,
                        int device_id);

// ---------------------------------------------------------------------------
// Runtime stats helpers
// ---------------------------------------------------------------------------

void recordCudaLinearCall(CudaModelWeights& storage);
void recordCudaLinearCopy(CudaModelWeights& storage, size_t h2d_bytes,
                          size_t d2h_bytes);
void recordCudaHostToDeviceCopy(CudaModelWeights& storage, size_t bytes);
void resetCudaRuntimeStats(CudaModelWeights& storage);

}  // namespace mini_llama

#endif  // MINI_LLAMA_CUDA_WEIGHT_UPLOAD_H_
