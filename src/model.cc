// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//

#include "mini_llama/model.h"

#include <stdexcept>

#include "mini_llama/quant.h"

namespace mini_llama {

static size_t tensorBytes(const Tensor& t) {
    return t.data.size() * sizeof(float);
}

static size_t quantizedTensorBytes(const QuantizedTensor& qt) {
    switch (qt.type) {
        case QuantType::kF32:
            return qt.f32_data.size() * sizeof(float);
        case QuantType::kQ80:
            return qt.q8_0_data.size() * sizeof(BlockQ80);
        case QuantType::kQ40:
            return qt.q4_0_data.size() * sizeof(BlockQ40);
        case QuantType::kQ41:
            return qt.q4_1_data.size() * sizeof(BlockQ41);
    }
    return 0;
}

static size_t quantizedTensorBytesF32(const QuantizedTensor& qt) {
    return qt.num_elements() * sizeof(float);
}

size_t modelWeightBytes(const MiniLlamaModel& model) {
    size_t bytes = 0;
    bytes += tensorBytes(model.token_embedding);
    bytes += tensorBytes(model.final_norm);
    bytes += quantizedTensorBytes(model.lm_head);
    for (const auto& lw : model.layers) {
        bytes += tensorBytes(lw.attention_norm);
        bytes += quantizedTensorBytes(lw.wq);
        bytes += quantizedTensorBytes(lw.wk);
        bytes += quantizedTensorBytes(lw.wv);
        bytes += tensorBytes(lw.bq);
        bytes += tensorBytes(lw.bk);
        bytes += tensorBytes(lw.bv);
        bytes += quantizedTensorBytes(lw.wo);
        bytes += tensorBytes(lw.ffn_norm);
        bytes += quantizedTensorBytes(lw.w_gate);
        bytes += quantizedTensorBytes(lw.w_up);
        bytes += quantizedTensorBytes(lw.w_down);
    }
    return bytes;
}

size_t modelWeightBytesF32(const MiniLlamaModel& model) {
    size_t bytes = 0;
    bytes += tensorBytes(model.token_embedding);
    bytes += tensorBytes(model.final_norm);
    bytes += quantizedTensorBytesF32(model.lm_head);
    for (const auto& lw : model.layers) {
        bytes += tensorBytes(lw.attention_norm);
        bytes += quantizedTensorBytesF32(lw.wq);
        bytes += quantizedTensorBytesF32(lw.wk);
        bytes += quantizedTensorBytesF32(lw.wv);
        bytes += tensorBytes(lw.bq);
        bytes += tensorBytes(lw.bk);
        bytes += tensorBytes(lw.bv);
        bytes += quantizedTensorBytesF32(lw.wo);
        bytes += tensorBytes(lw.ffn_norm);
        bytes += quantizedTensorBytesF32(lw.w_gate);
        bytes += quantizedTensorBytesF32(lw.w_up);
        bytes += quantizedTensorBytesF32(lw.w_down);
    }
    return bytes;
}

static Tensor quantizedTensorToF32(const QuantizedTensor& qt) {
    switch (qt.type) {
        case QuantType::kF32:
            return toTensor(qt);
        case QuantType::kQ80:
            return dequantizeFromQ80(qt.q8_0_data, qt.shape);
        case QuantType::kQ40:
            return dequantizeFromQ40(qt.q4_0_data, qt.shape);
        case QuantType::kQ41:
            return dequantizeFromQ41(qt.q4_1_data, qt.shape);
    }
    BIZLOG(ErrorCode::kQuantError,
           "quantized_tensor_to_f32: unknown quant type");
    return Tensor();
}

static void quantizeQtToQ80(QuantizedTensor& qt) {
    if (qt.type == QuantType::kQ80) {
        return;
    }
    Tensor t = quantizedTensorToF32(qt);
    qt.q8_0_data = quantizeToQ80(t);
    qt.type = QuantType::kQ80;
    qt.f32_data.clear();
    qt.f32_data.shrink_to_fit();
    qt.q4_0_data.clear();
    qt.q4_0_data.shrink_to_fit();
    qt.q4_1_data.clear();
    qt.q4_1_data.shrink_to_fit();
}

void quantizeModelToQ80(MiniLlamaModel& model) {
    quantizeQtToQ80(model.lm_head);
    for (auto& lw : model.layers) {
        quantizeQtToQ80(lw.wq);
        quantizeQtToQ80(lw.wk);
        quantizeQtToQ80(lw.wv);
        quantizeQtToQ80(lw.wo);
        quantizeQtToQ80(lw.w_gate);
        quantizeQtToQ80(lw.w_up);
        quantizeQtToQ80(lw.w_down);
    }
}

static void quantizeQtToQ40(QuantizedTensor& qt) {
    if (qt.type == QuantType::kQ40) {
        return;
    }
    Tensor t = quantizedTensorToF32(qt);
    qt.q4_0_data = quantizeToQ40(t);
    qt.type = QuantType::kQ40;
    qt.f32_data.clear();
    qt.f32_data.shrink_to_fit();
    qt.q8_0_data.clear();
    qt.q8_0_data.shrink_to_fit();
    qt.q4_1_data.clear();
    qt.q4_1_data.shrink_to_fit();
}

void quantizeModelToQ40(MiniLlamaModel& model) {
    quantizeQtToQ40(model.lm_head);
    for (auto& lw : model.layers) {
        quantizeQtToQ40(lw.wq);
        quantizeQtToQ40(lw.wk);
        quantizeQtToQ40(lw.wv);
        quantizeQtToQ40(lw.wo);
        quantizeQtToQ40(lw.w_gate);
        quantizeQtToQ40(lw.w_up);
        quantizeQtToQ40(lw.w_down);
    }
}

void uploadModelWeightsToCuda(MiniLlamaModel& model, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    if (!model.loaded) {
        throw std::runtime_error("CUDA weight upload requires a loaded model");
    }

    cudaSetDeviceId(device_id);
    auto storage = std::make_shared<CudaModelWeights>();
    storage->device_id = device_id;

    // Global weights.
    uploadF32TensorWeight(*storage, model.token_embedding, "token_embedding",
                          device_id);
    uploadF32TensorWeight(*storage, model.final_norm, "final_norm", device_id);
    uploadLinearWeight(*storage, model.lm_head, "lm_head", device_id);

    // Per-layer weights.
    for (size_t layer = 0; layer < model.layers.size(); ++layer) {
        const LayerWeights& lw = model.layers[layer];
        const std::string prefix = "layers." + std::to_string(layer) + ".";

        // Attention.
        uploadF32TensorWeight(*storage, lw.attention_norm,
                              prefix + "attention_norm", device_id);
        uploadLinearWeight(*storage, lw.wq, prefix + "wq", device_id);
        uploadLinearWeight(*storage, lw.wk, prefix + "wk", device_id);
        uploadLinearWeight(*storage, lw.wv, prefix + "wv", device_id);
        uploadF32TensorWeight(*storage, lw.bq, prefix + "bq", device_id);
        uploadF32TensorWeight(*storage, lw.bk, prefix + "bk", device_id);
        uploadF32TensorWeight(*storage, lw.bv, prefix + "bv", device_id);
        uploadLinearWeight(*storage, lw.wo, prefix + "wo", device_id);

        // FFN.
        uploadF32TensorWeight(*storage, lw.ffn_norm, prefix + "ffn_norm",
                              device_id);
        uploadLinearWeight(*storage, lw.w_gate, prefix + "w_gate", device_id);
        uploadLinearWeight(*storage, lw.w_up, prefix + "w_up", device_id);
        uploadLinearWeight(*storage, lw.w_down, prefix + "w_down", device_id);
    }

    model.cuda_weights = std::move(storage);
#else
    (void)model;
    (void)device_id;
    throw std::runtime_error(
        "CUDA weight upload requires CUDA. Reconfigure with "
        "-DMINI_LLAMA_CUDA=ON.");
#endif
}

void clearModelCudaWeights(MiniLlamaModel& model) {
    model.cuda_weights.reset();
}

bool modelHasCudaWeights(const MiniLlamaModel& model) {
    return model.cuda_weights != nullptr;
}

size_t modelCudaUploadedWeightCount(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->uploaded_weight_count;
}

size_t modelCudaMemoryBytes(const MiniLlamaModel& model) {
    if (!model.cuda_weights) {
        return 0;
    }
    return model.cuda_weights->uploaded_bytes;
}

}  // namespace mini_llama
