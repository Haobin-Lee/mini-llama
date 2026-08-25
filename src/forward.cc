// Copyright (c) 2026
// SPDX-License-Identifier: MIT

//   embed_token(model, token) -> [dim]
//   forward_layer(ctx, model, layer, x, pos):
//     rmsnorm -> q/k/v projection -> rope -> attention (KV cache) -> residual
//     -> rmsnorm -> ffn (SwiGLU) -> residual
//   compute_logits(model, x) -> [vocab_size]
// ForwardToken runs one token; ForwardBatch loops ForwardToken over the batch.

#include "mini_llama/forward.h"

#include <iostream>
#include <stdexcept>

#include "errlog/bizlog.h"
#include "mini_llama/cuda_attention.h"
#include "mini_llama/cuda_matmul.h"
#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_quant.h"
#include "mini_llama/ops.h"
#include "mini_llama/threadpool.h"

namespace mini_llama {

using errlog::ErrorCode;
static bool validateForwardInputs(const MiniLlamaContext& ctx,
                                  const MiniLlamaModel& model, int token) {
    const ModelConfig& c = model.config;
    if (!model.loaded) {
        BIZLOG(ErrorCode::kForwardFailed,
               "ForwardToken called with an unloaded model");
        return false;
    }
    if (token < 0 || token >= c.vocab_size) {
        BIZLOG(ErrorCode::kForwardFailed,
               "ForwardToken token id out of range, token: " +
                   std::to_string(token) +
                   ", vocab_size: " + std::to_string(c.vocab_size));
        return false;
    }
    if (ctx.pos < 0 || ctx.pos >= c.max_seq_len) {
        BIZLOG(ErrorCode::kForwardFailed, "ForwardToken position out of range");
        return false;
    }
    if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
        BIZLOG(ErrorCode::kForwardFailed,
               "ForwardToken layer count does not match model config");
        return false;
    }
    if (model.token_embedding.data.size() <
        static_cast<size_t>(c.vocab_size * c.dim)) {
        BIZLOG(ErrorCode::kForwardFailed,
               "ForwardToken token embedding tensor is smaller than config");
        return false;
    }

    return true;
}

// ===========================================================================
// CUDA weight lookup and profiling
// ===========================================================================
static const CudaUploadedWeight* findCudaWeight(const CudaModelWeights& storage,
                                                const std::string& name,
                                                QuantType type) {
    for (const auto& weight : storage.weights) {
        if (weight.name == name && weight.type == type) {
            return &weight;
        }
    }
    return nullptr;
}

int cudaDeviceId(const MiniLlamaModel& model) {
    return model.cuda_weights ? model.cuda_weights->device_id : 0;
}

bool cudaLinearSupported(const QuantizedTensor& weight) {
    return weight.type == QuantType::kF32 || weight.type == QuantType::kQ80 ||
           weight.type == QuantType::kQ40 || weight.type == QuantType::kQ41;
}

bool cudaLinearReady(const MiniLlamaModel& model, const std::string& name,
                     const QuantizedTensor& weight) {
    return model.cuda_weights && cudaLinearSupported(weight) &&
           findCudaWeight(*model.cuda_weights, name, weight.type) != nullptr;
}

// Profiling recorders (no-op if no cuda_weights).
void recordLinearCall(const MiniLlamaModel& model) {
    if (model.cuda_weights) {
        model.cuda_weights->linear_calls += 1;
    }
}

void recordLinearCopy(const MiniLlamaModel& model, const Tensor& x,
                      const Tensor& y) {
    if (!model.cuda_weights) {
        return;
    }
    auto& s = *model.cuda_weights;
    s.linear_calls += 1;
    s.host_to_device_copies += 1;
    s.device_to_host_copies += 1;
    s.host_to_device_bytes += x.size() * sizeof(float);
    s.device_to_host_bytes += y.size() * sizeof(float);
}

void recordActivationCall(const MiniLlamaModel& model, size_t n = 1) {
    if (model.cuda_weights) {
        model.cuda_weights->activation_calls += n;
    }
}

void recordAttentionCall(const MiniLlamaModel& model) {
    if (model.cuda_weights) {
        model.cuda_weights->attention_calls += 1;
    }
}

void recordKvCacheWrite(const MiniLlamaModel& model, size_t bytes) {
    if (model.cuda_weights) {
        model.cuda_weights->kv_cache_write_bytes += bytes;
    }
}

void recordKvCacheRead(const MiniLlamaModel& model, size_t bytes) {
    if (model.cuda_weights) {
        model.cuda_weights->kv_cache_read_bytes += bytes;
    }
}

void recordHostToDeviceCopy(const MiniLlamaModel& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->host_to_device_copies += 1;
    model.cuda_weights->host_to_device_bytes += bytes;
}

void recordDeviceToHostCopy(const MiniLlamaModel& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->device_to_host_copies += 1;
    model.cuda_weights->device_to_host_bytes += bytes;
}

CudaTensor uploadCudaTensor(const MiniLlamaModel& model, const Tensor& x) {
    CudaTensor x_dev = cudaTensorFromHost(x, cudaDeviceId(model));
    recordHostToDeviceCopy(model, x.size() * sizeof(float));
    return x_dev;
}

Tensor downloadCudaTensor(const MiniLlamaModel& model, const CudaTensor& x) {
    Tensor y = x.download();
    recordDeviceToHostCopy(model, y.size() * sizeof(float));
    return y;
}

// ===========================================================================
// Embedding lookup
// ===========================================================================

static Tensor embedToken(const MiniLlamaModel& model, int token_id) {
    int dim = model.config.dim;
    Tensor x({dim}, 0.0f);
    for (int i = 0; i < dim; ++i) {
        x.data[i] = model.token_embedding.data[token_id * dim + i];
    }
    return x;
}

static CudaTensor embedTokenDevice(const MiniLlamaModel& model, int token_id) {
    const CudaUploadedWeight* cw =
        findCudaWeight(*model.cuda_weights, "token_embedding", QuantType::kF32);
    if (cw == nullptr || cw->linear) {
        throw std::runtime_error(
            "CUDA forward embedding missing uploaded weight: token_embedding");
    }
    return cudaEmbeddingLookupDeviceWeight(cw->buffer.data(), cw->shape,
                                           token_id, cudaDeviceId(model));
}

// ===========================================================================
// Linear layer dispatch (F32 / Q8_0 / Q4_0 / Q4_1)
// ===========================================================================
// Device input → Device output (zero-copy pipeline).
CudaTensor forwardLinearDevice(const MiniLlamaModel& model,
                               const std::string& name, const CudaTensor& x,
                               const QuantizedTensor& weight) {
    if (!cudaLinearSupported(weight)) {
        throw std::runtime_error(
            "CUDA forward linear unsupported weight type: " + name);
    }
    const CudaUploadedWeight* cw =
        findCudaWeight(*model.cuda_weights, name, weight.type);
    if (cw == nullptr) {
        throw std::runtime_error(
            "CUDA forward linear missing uploaded weight: " + name);
    }

    int dev = cudaDeviceId(model);
    CudaTensor y;
    switch (weight.type) {
        case QuantType::kF32:
            y = cudaLinearDeviceInput(x, cw->buffer.data(), cw->shape, dev);
            break;
        case QuantType::kQ80:
            y = cudaQ80LinearDeviceInput(x, cw->buffer.data(), cw->block_count,
                                         cw->shape, dev);
            break;
        case QuantType::kQ40:
            y = cudaQ40LinearDeviceInput(x, cw->buffer.data(), cw->block_count,
                                         cw->shape, dev);
            break;
        case QuantType::kQ41:
            y = cudaQ41LinearDeviceInput(x, cw->buffer.data(), cw->block_count,
                                         cw->shape, dev);
            break;
    }
    recordLinearCall(model);
    return y;
}

static Tensor forwardLinear(const MiniLlamaModel& model, const Tensor& input,
                            const QuantizedTensor& weight,
                            std::string_view name) {
    if (!model.cuda_weights || !cudaLinearSupported(weight)) {
        return linear(input, weight);
    }
    const CudaUploadedWeight* cw =
        findCudaWeight(*model.cuda_weights, name.data(), weight.type);
    if (cw == nullptr) {
        throw std::runtime_error(
            "CUDA forward linear missing uploaded weight: " +
            std::string(name));
    }

    int dev = cudaDeviceId(model);
    Tensor y;
    switch (weight.type) {
        case QuantType::kF32:
            y = cudaLinearDeviceWeight(input, cw->buffer.data(), cw->shape,
                                       nullptr, dev);
            break;
        case QuantType::kQ80:
            y = cudaQ80LinearDeviceWeight(input, cw->buffer.data(),
                                          cw->block_count, cw->shape, nullptr,
                                          dev);
            break;
        case QuantType::kQ40:
            y = cudaQ40LinearDeviceWeight(input, cw->buffer.data(),
                                          cw->block_count, cw->shape, nullptr,
                                          dev);
            break;
        case QuantType::kQ41:
            y = cudaQ41LinearDeviceWeight(input, cw->buffer.data(),
                                          cw->block_count, cw->shape, nullptr,
                                          dev);
            break;
    }
    recordLinearCopy(model, input, y);
    return y;
}

static bool checkShape(const Tensor& t, const std::vector<int>& expected,
                       const char* caller) {
    return t.isSameShape(expected, caller);
}

// ===========================================================================
// Sub-operations: RmsNorm, Add, Silu, RoPE, Softmax, Embedding
// ===========================================================================

CudaTensor forwardRmsNormDevice(const MiniLlamaModel& model,
                                const std::string& name, const CudaTensor& x,
                                const Tensor& weight, float eps) {
    const CudaUploadedWeight* cw =
        findCudaWeight(*model.cuda_weights, name, QuantType::kF32);
    if (cw == nullptr || cw->linear) {
        throw std::runtime_error(
            "CUDA forward RMSNorm missing uploaded weight: " + name);
    }
    if (cw->shape != weight.shape) {
        throw std::runtime_error(
            "CUDA forward RMSNorm uploaded weight shape mismatch: " + name);
    }
    CudaTensor y = cudaRmsNormDeviceWeight(x, cw->buffer.data(), cw->shape, eps,
                                           cudaDeviceId(model));
    recordActivationCall(model);
    return y;
}
static Tensor forwardRmsNorm(const MiniLlamaModel& model, const Tensor& x,
                             const Tensor& weight, float eps) {
    if (!model.cuda_weights) {
        return rmsNorm(x, weight, eps);
    }
    Tensor y = cudaRmsNorm(x, weight, eps, cudaDeviceId(model));
    recordActivationCall(model);
    return y;
}

static CudaTensor forwardAddDevice(const MiniLlamaModel& model,
                                   const CudaTensor& a, const CudaTensor& b) {
    CudaTensor y = cudaElementwiseAddDeviceInput(a, b, cudaDeviceId(model));
    recordActivationCall(model);
    return y;
}

static Tensor forwardAdd(const MiniLlamaModel& model, const Tensor& a,
                         const Tensor& b) {
    if (!model.cuda_weights) {
        if (a.shape != b.shape) {
            BIZLOG(ErrorCode::kOpShapeMismatch, "forwardAdd", a.shapeString(),
                   b.shapeString());
            return Tensor();
        }
        Tensor sum{a.shape, 0.0f};
        for (size_t i = 0; i < a.size(); ++i) {
            sum.data[i] = a.data[i] + b.data[i];
        }
        return sum;
    }
    Tensor sum = cudaElementwiseAdd(a, b, cudaDeviceId(model));
    recordActivationCall(model);
    return sum;
}

static void forwardRopeDevice(const MiniLlamaModel& model, CudaTensor& q,
                              CudaTensor& k, int n_heads, int n_kv_heads,
                              int head_dim, int pos, float theta,
                              RopeType rope_type) {
    cudaRopeDeviceInput(q, k, n_heads, n_kv_heads, head_dim, pos, theta,
                        rope_type, cudaDeviceId(model));
    recordActivationCall(model);
}

static bool forwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k,
                        int pos, float theta, RopeType rope_type) {
    if (!model.cuda_weights) {
        return rope(q, k, pos, theta, rope_type);
    }
    cudaRope(q, k, pos, theta, rope_type, cudaDeviceId(model));
    recordActivationCall(model);

    return true;
}

static Tensor forwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
    if (!model.cuda_weights) {
        return softmax(x);
    }

    Tensor y = cudaSoftmax(x, cudaDeviceId(model));
    recordActivationCall(model);
    return y;
}

// SwiGLU: silu(gate) * up
static CudaTensor forwardSwiGluDevice(const MiniLlamaModel& model,
                                      const CudaTensor& gate,
                                      const CudaTensor& up) {
    CudaTensor silu_gate = cudaSiluDeviceInput(gate, cudaDeviceId(model));
    CudaTensor y =
        cudaElementwiseMulDeviceInput(silu_gate, up, cudaDeviceId(model));
    recordActivationCall(model, 2);
    return y;
}

static Tensor forwardSiGlu(const MiniLlamaModel& model, const Tensor& gate,
                           const Tensor& up) {
    if (!model.cuda_weights) {
        return swiGlu(gate, up);
    }
    Tensor silu_gate = cudaSilu(gate, cudaDeviceId(model));
    Tensor y = cudaElementwiseMul(silu_gate, up, cudaDeviceId(model));
    recordActivationCall(model, 2);
    return y;
}

// Device bias addition using pre-uploaded weight.
CudaTensor addOptionalBiasDevice(const MiniLlamaModel& model,
                                 const std::string& name, CudaTensor x,
                                 const Tensor& bias) {
    if (bias.data.empty()) {
        return x;
    }
    const CudaUploadedWeight* cw =
        findCudaWeight(*model.cuda_weights, name, QuantType::kF32);
    if (cw == nullptr || cw->linear) {
        throw std::runtime_error("CUDA forward bias missing uploaded weight: " +
                                 name);
    }
    CudaTensor y = cudaElementwiseAddDeviceWeight(
        x, cw->buffer.data(), cw->shape, cudaDeviceId(model));
    recordActivationCall(model);
    return y;
}

static Tensor addOptionalBias(const Tensor& x, const Tensor& bias,
                              const char* caller) {
    if (bias.data.empty()) {
        return x;
    }
    if (x.numDims() != 1 || bias.numDims() != 1 ||
        x.shape[0] != bias.shape[0]) {
        BIZLOG(ErrorCode::kForwardBiasMismatch, caller, x.shapeString(),
               bias.shapeString());
        return Tensor();
    }

    Tensor y = x;
    for (int i = 0; i < x.shape[0]; ++i) {
        y.data[i] += bias.data[i];
    }
    return y;
}

// ===========================================================================
// QKV Projection
// ===========================================================================

void forwardQkvProjectionDevice(const MiniLlamaModel& model,
                                const std::string& prefix, const CudaTensor& h,
                                const LayerWeights& lw, CudaTensor& q,
                                CudaTensor& k, CudaTensor& v) {
    q = forwardLinearDevice(model, prefix + "wq", h, lw.wq);
    k = forwardLinearDevice(model, prefix + "wk", h, lw.wk);
    v = forwardLinearDevice(model, prefix + "wv", h, lw.wv);
    q = addOptionalBiasDevice(model, prefix + "bq", std::move(q), lw.bq);
    k = addOptionalBiasDevice(model, prefix + "bk", std::move(k), lw.bk);
    v = addOptionalBiasDevice(model, prefix + "bv", std::move(v), lw.bv);
}

static void forwardQKVProjection(const MiniLlamaModel& model,
                                 std::string_view layer_prefix,
                                 const Tensor& input,
                                 const LayerWeights& layer_weights,
                                 Tensor& q_flat, Tensor& k_flat,
                                 Tensor& v_flat) {
    // TODO: cuda q/k/v projection

    std::string_view q_name = std::string(layer_prefix) + "wq";
    std::string_view k_name = std::string(layer_prefix) + "wk";
    std::string_view v_name = std::string(layer_prefix) + "wv";

    q_flat =
        addOptionalBias(forwardLinear(model, input, layer_weights.wq, q_name),
                        layer_weights.bq, "forward layer q");
    k_flat =
        addOptionalBias(forwardLinear(model, input, layer_weights.wk, k_name),
                        layer_weights.bk, "forward layer k");
    v_flat =
        addOptionalBias(forwardLinear(model, input, layer_weights.wv, v_name),
                        layer_weights.bv, "forward layer v");
}

// ===========================================================================
// Attention
// ===========================================================================

static int mapKVHead(int n_heads, int n_kv_heads, int n) {
    return n / (n_heads / n_kv_heads);
}

static CudaTensor attentionForwardDevice(
    const MiniLlamaModel& model, const CudaTensor& q, const CudaTensor& k,
    const CudaTensor& v, int pos, int layer, CudaKvCache& cuda_kv_cache,
    int n_heads, int n_kv_heads, int head_dim) {
    cuda_kv_cache.writeDevice(layer, pos, k, v);
    recordKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));

    CudaTensor out = cudaAttentionDecodeDeviceInput(
        q, cuda_kv_cache, layer, pos, n_heads, n_kv_heads, head_dim,
        cudaDeviceId(model));
    recordAttentionCall(model);
    recordKvCacheRead(
        model, static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) *
                   static_cast<size_t>(head_dim) * sizeof(float) * 2);
    return out;
}

static Tensor forwardAttention(KvCache& kv_cache, int pos,
                               const MiniLlamaModel& model,
                               CudaKvCache* cuda_kv_cache, const Tensor& q,
                               const Tensor& k, const Tensor& v, int layer) {
    const ModelConfig& model_config = model.config;
    int n_kv_heads = model_config.n_kv_heads;
    int n_heads = model_config.n_heads;
    int head_dim = model_config.head_dim;
    if (cuda_kv_cache != nullptr && !cuda_kv_cache->empty()) {
        cuda_kv_cache->write(layer, pos, k, v);
        recordKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
        Tensor out =
            cudaAttentionDecode(q, *cuda_kv_cache, layer, pos, n_heads,
                                n_kv_heads, head_dim, cudaDeviceId(model));
        recordAttentionCall(model);
        recordKvCacheRead(
            model, static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) *
                       static_cast<size_t>(head_dim) * sizeof(float) * 2);
        recordHostToDeviceCopy(model, q.size() * sizeof(float));
        recordDeviceToHostCopy(model, out.size() * sizeof(float));
        return out;
    }

    // 缓存KV
    kv_cache.write(layer, pos, k, v);
    int kv_dim = n_kv_heads * head_dim;
    if (cuda_kv_cache != nullptr && !cuda_kv_cache->empty()) {
        cuda_kv_cache->write(layer, pos, k, v);
        recordKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
        Tensor out =
            cudaAttentionDecode(q, *cuda_kv_cache, layer, pos, n_heads,
                                n_kv_heads, head_dim, cudaDeviceId(model));
        recordAttentionCall(model);
        recordKvCacheRead(
            model, static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) *
                       static_cast<size_t>(head_dim) * sizeof(float) * 2);
        recordHostToDeviceCopy(model, q.size() * sizeof(float));
        recordDeviceToHostCopy(model, out.size() * sizeof(float));
        return out;
    }

    float scale = std::sqrt(1.0f / head_dim);
    Tensor attention_out{{n_heads, head_dim}, 0.0f};
    auto compute_head = [&](int head) {
        float sum = 0.0f;
        int kv_head = mapKVHead(n_heads, n_kv_heads, head);
        std::vector<float> scores_data(pos + 1, 0.0f);
        for (int i = 0; i <= pos; ++i) {
            sum = 0.0f;
            auto* key_ptr = kv_cache.keyPtr(layer, i, kv_head);
            if (key_ptr == nullptr) {
                return;
            }
            for (int j = 0; j < head_dim; ++j) {
                sum += q.data[head * head_dim + j] * key_ptr[j];
            }
            scores_data[i] = sum * scale;
        }

        Tensor scores({pos + 1}, 0.0f);
        scores.data = std::move(scores_data);
        Tensor probs = forwardSoftmax(model, scores);

        for (int i = 0; i < head_dim; ++i) {
            sum = 0.0f;
            for (int j = 0; j <= pos; ++j) {
                auto* v_ptr = kv_cache.valuePtr(layer, j, kv_head);
                if (v_ptr == nullptr) {
                    return;
                }
                sum += probs.data[j] * v_ptr[i];
            }
            attention_out.data[head * head_dim + i] = sum;
        }
    };

    ThreadPool::submitTask(n_heads, [&](int begin, int end) {
        for (int h = begin; h < end; ++h) {
            compute_head(h);
        }
    });

    return attention_out;
}

// ===========================================================================
// FFN (SwiGLU)
// ===========================================================================

CudaTensor ffnForwardDevice(const MiniLlamaModel& model,
                            const std::string& prefix, const CudaTensor& h,
                            const LayerWeights& lw) {
    CudaTensor gate =
        forwardLinearDevice(model, prefix + "w_gate", h, lw.w_gate);
    CudaTensor up = forwardLinearDevice(model, prefix + "w_up", h, lw.w_up);
    CudaTensor ff = forwardSwiGluDevice(model, gate, up);
    return forwardLinearDevice(model, prefix + "w_down", ff, lw.w_down);
}

static Tensor ffnForward(const MiniLlamaModel& model,
                         std::string_view layer_prefix, const Tensor& input,
                         const LayerWeights& layer_weights) {
    std::string_view gate_name = std::string(layer_prefix) + "w_gate";
    std::string_view up_name = std::string(layer_prefix) + "w_up";
    std::string_view down_name = std::string(layer_prefix) + "w_down";

    Tensor gate = forwardLinear(model, input, layer_weights.w_gate,
                                gate_name);  // [hidden_dim]
    Tensor up = forwardLinear(model, input, layer_weights.w_up,
                              up_name);  // [hidden_dim]

    // 动态决定gate up分支保留多少特征
    Tensor ff = forwardLinear(model, forwardSiGlu(model, gate, up),
                              layer_weights.w_down, down_name);

    return ff;
}

// ===========================================================================
// Single Transformer layer
// ===========================================================================

CudaTensor forwardLayerDevice(MiniLlamaContext& ctx,
                              const MiniLlamaModel& model, const CudaTensor& x,
                              int layer, const LayerWeights& lw,
                              const ModelConfig& c) {
    int dim = c.dim;
    int n_heads = c.n_heads;
    int n_kv_heads = c.n_kv_heads;
    int head_dim = c.head_dim;
    int pos = ctx.pos;
    std::string prefix = "layers." + std::to_string(layer) + ".";

    // ---- Attention sublayer ----
    CudaTensor h = forwardRmsNormDevice(model, prefix + "attention_norm", x,
                                        lw.attention_norm, c.rms_norm_eps);

    CudaTensor q, k, v;
    forwardQkvProjectionDevice(model, prefix, h, lw, q, k, v);
    forwardRopeDevice(model, q, k, n_heads, n_kv_heads, head_dim, pos,
                      c.rope_theta, c.rope_type);

    CudaTensor attn_out =
        attentionForwardDevice(model, q, k, v, pos, layer, ctx.cuda_kv_cache,
                               n_heads, n_kv_heads, head_dim);

    CudaTensor attn_proj =
        forwardLinearDevice(model, prefix + "wo", attn_out, lw.wo);
    CudaTensor x_attn = forwardAddDevice(model, x, attn_proj);

    // ---- FFN sublayer ----
    CudaTensor h2 = forwardRmsNormDevice(model, prefix + "ffn_norm", x_attn,
                                         lw.ffn_norm, c.rms_norm_eps);
    CudaTensor ff = ffnForwardDevice(model, prefix, h2, lw);
    return forwardAddDevice(model, x_attn, ff);
}

static Tensor forwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                           const Tensor& input, int layer) {
    const ModelConfig& model_config = model.config;
    const LayerWeights& layer_weights = model.layers[layer];
    int n_heads = model_config.n_heads;
    int n_kv_heads = model_config.n_kv_heads;
    int head_dim = model_config.head_dim;
    int dim = model_config.dim;
    int pos = ctx.pos;
    std::string layer_prefix = "layer" + std::to_string(layer) + ".";

    // 1. RMS Norm
    Tensor norm_input = forwardRmsNorm(
        model, input, layer_weights.attention_norm, model_config.rms_norm_eps);
    if (norm_input.empty()) {
        std::cout << "norm_input empty" << std::endl;
        return norm_input;
    }

    // 2. QKV projection
    Tensor q_flat;
    Tensor k_flat;
    Tensor v_flat;
    forwardQKVProjection(model, layer_prefix, norm_input, layer_weights, q_flat,
                         k_flat, v_flat);

    // todo reshape
    q_flat = q_flat.reshapeChecked({n_heads, head_dim}, "forward layer q");
    k_flat = k_flat.reshapeChecked({n_kv_heads, head_dim}, "forward layer k");
    v_flat = v_flat.reshapeChecked({n_kv_heads, head_dim}, "forward layer v");

    if (!forwardRope(model, q_flat, k_flat, pos, model_config.rope_theta,
                     model_config.rope_type)) {
        std::cout << " forwardRope error" << std::endl;
        return Tensor();
    }

    // 3. Attention
    Tensor attention_out =
        forwardAttention(ctx.kv_cache, pos, model,
                         model.cuda_weights ? &ctx.cuda_kv_cache : nullptr,
                         q_flat, k_flat, v_flat, layer);
    if (attention_out.empty()) {
        std::cout << "attention_out empty" << std::endl;
        return attention_out;
    }

    // 4. reshape, projection, residual
    Tensor attention_conn = attention_out.reshapeChecked(
        {1, n_heads * head_dim}, "forward layer attention_out");
    if (attention_conn.empty()) {
        std::cout << "attention_conn empty" << std::endl;
        return attention_conn;
    }
    Tensor attention_proj =
        forwardLinear(model, attention_conn, layer_weights.wo,
                      std::string(layer_prefix) + "wo");
    if (!checkShape(attention_proj, {1, dim},
                    "forward layer attention projection")) {
        return Tensor();
    }
    Tensor attention_proj_flat = attention_proj.reshapeChecked(
        {dim}, "forward layer attention projection flat");
    if (attention_proj_flat.empty()) {
        std::cout << "attention_proj_flat empty" << std::endl;
        return attention_proj_flat;
    }
    Tensor attention_residual = forwardAdd(model, input, attention_proj_flat);
    if (attention_residual.empty()) {
        std::cout << "attention_residual empty" << std::endl;
        return attention_residual;
    }

    // 5. FFN
    Tensor ffn_norm =
        forwardRmsNorm(model, attention_residual, layer_weights.ffn_norm,
                       model_config.rms_norm_eps);
    if (ffn_norm.empty()) {
        std::cout << "ffn_norm empty" << std::endl;
        return ffn_norm;
    }
    Tensor ffn_out = ffnForward(model, layer_prefix, ffn_norm, layer_weights);
    if (!checkShape(ffn_out, {dim}, "forward layer ffn")) {
        std::cout << "ffn_out shape error" << std::endl;
        return Tensor();
    }

    return forwardAdd(model, attention_residual, ffn_out);
}

// ===========================================================================
// Compute logits from final hidden state
// ===========================================================================

static Tensor computeLogitsDevice(const CudaTensor& x,
                                  const MiniLlamaModel& model,
                                  const ModelConfig& c) {
    CudaTensor normed = forwardRmsNormDevice(model, "final_norm", x,
                                             model.final_norm, c.rms_norm_eps);
    CudaTensor logits_dev =
        forwardLinearDevice(model, "lm_head", normed, model.lm_head);
    Tensor logits = downloadCudaTensor(model, logits_dev);
    return logits.reshapeChecked({c.vocab_size}, "computeLogitsDevice");
}

static Tensor computelogits(const MiniLlamaModel& model, const Tensor& x) {
    const ModelConfig& model_config = model.config;
    Tensor norm_tensor =
        forwardRmsNorm(model, x, model.final_norm, model_config.rms_norm_eps);
    if (norm_tensor.empty()) {
        return norm_tensor;
    }
    Tensor logits_flat =
        forwardLinear(model, norm_tensor, model.lm_head, "lm_head");
    if (logits_flat.empty()) {
        return logits_flat;
    }
    return logits_flat.reshapeChecked({model_config.vocab_size},
                                      "computeLogits");
}

Tensor forwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token) {
    if (!validateForwardInputs(ctx, model, token)) {
        return Tensor();
    }

    const ModelConfig& c = model.config;
    int n_layers = c.n_layers;
    if (model.cuda_weights && ctx.cuda_kv_cache.empty()) {
        ctx.cuda_kv_cache.reset(c.n_layers, c.max_seq_len, c.n_kv_heads,
                                c.head_dim, model.cuda_weights->device_id);
    }

    // 1. Embedding lookup
    if (model.cuda_weights) {
        CudaTensor x_dev = embedTokenDevice(model, token);
        for (int layer = 0; layer < n_layers; ++layer) {
            x_dev = forwardLayerDevice(ctx, model, x_dev, layer,
                                       model.layers[layer], c);
        }
        return computeLogitsDevice(x_dev, model, c);
    }

    Tensor x = embedToken(model, token);
    // 2. Transformer layers
    for (int layer = 0; layer < n_layers; ++layer) {
        x = forwardLayer(ctx, model, x, layer);
        if (x.empty()) {
            return x;
        }
    }
    // 3. Norm + logits
    return computelogits(model, x);
}

}  // namespace mini_llama
