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

static Tensor embedToken(const MiniLlamaModel& model, int token_id) {
    int dim = model.config.dim;
    Tensor x({dim}, 0.0f);
    // 拿到第token_id行的向量即为token的embeded向量
    for (int i = 0; i < dim; ++i) {
        x.data[i] = model.token_embedding.data[token_id * dim + i];
    }

    return x;
}

static Tensor forwardRmsNorm(const MiniLlamaModel& model, const Tensor& x,
                             const Tensor& weight, float eps) {
    // TODO: cuda rms norm
    return rmsNorm(x, weight, eps);
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

static Tensor forwardLinear(const MiniLlamaModel& model, const Tensor& input,
                            const QuantizedTensor& weight,
                            std::string_view name) {
    return linear(input, weight);

    // TODO: cuda linear forward
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

static bool checkShape(const Tensor& t, const std::vector<int>& expected,
                       const char* caller) {
    return t.isSameShape(expected, caller);
}

static Tensor forwardSiGlu(const MiniLlamaModel& model, const Tensor& gate,
                           const Tensor& up) {
    // TODO: cuda swiGLU
    return swiGlu(gate, up);
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

static Tensor forwardAdd(const MiniLlamaModel& model, const Tensor& a,
                         const Tensor& b) {
    // TODO: cuda add

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

static bool forwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k,
                        int pos, float theta, RopeType rope_type) {
    // TODO: cuda rope
    return rope(q, k, pos, theta, rope_type);
}

static int mapKVHead(int n_heads, int n_kv_heads, int n) {
    return n / (n_heads / n_kv_heads);
}

static Tensor forwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
    // TODO: cuda softmax
    return softmax(x);
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
                                      "compute_logits");
}

static Tensor forwardAttention(KvCache& kv_cache, int pos,
                               const MiniLlamaModel& model, const Tensor& q,
                               const Tensor& k, const Tensor& v, int layer) {
    // TODO: cuda attention
    const ModelConfig& model_config = model.config;
    int n_kv_heads = model_config.n_kv_heads;
    int n_heads = model_config.n_heads;
    int head_dim = model_config.head_dim;
    int kv_dim = n_kv_heads * head_dim;

    // 缓存KV
    kv_cache.write(layer, pos, k, v);

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

    // TODO: cuda compute head

    ThreadPool::submitTask(n_heads, [&](int begin, int end) {
        for (int h = begin; h < end; ++h) {
            compute_head(h);
        }
    });

    return attention_out;
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
    Tensor attention_out = forwardAttention(ctx.kv_cache, pos, model, q_flat,
                                            k_flat, v_flat, layer);
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

Tensor forwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token) {
    if (!validateForwardInputs(ctx, model, token)) {
        return Tensor();
    }

    const ModelConfig& c = model.config;
    int n_layers = c.n_layers;

    // TODO: init cuda kv cache

    // TODO: cuda forward

    // 1. Token embedding
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
