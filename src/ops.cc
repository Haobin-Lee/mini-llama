// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/ops.h"

#include <cmath>
#include <stdexcept>

#include "errlog/bizlog.h"  // 错误码日志：BIZLOG / ErrorCode
#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/quant.h"

namespace mini_llama {

using errlog::ErrorCode;

Tensor matmul(const Tensor& a, const Tensor& b) {
    return matmulDispatch(a, b, defaultMatmulMode());
}

Tensor linear(const Tensor& x, const Tensor& weight) {
    return linearDispatch(x, weight, defaultMatmulMode());
}

// ---------------------------------------------------------------------------
// Quantized Linear dispatch
// ---------------------------------------------------------------------------
Tensor linear(const Tensor& x, const QuantizedTensor& weight) {
    switch (weight.type) {
        case QuantType::kF32:
            return linear(x, toTensor(weight));
        case QuantType::kQ80:
            return linearQ80(x, weight.q8_0_data, weight.shape);
        case QuantType::kQ40:
            return linearQ40(x, weight.q4_0_data, weight.shape);
        case QuantType::kQ41:
            return linearQ41(x, weight.q4_1_data, weight.shape);
    }
    return Tensor();
}

// 均方根归一化，防止数据发散
Tensor rmsNorm(const Tensor& x, const Tensor& weight, float eps) {
    if (x.numDims() != 1 || weight.numDims() != 1) {
        // 错误码：算子输入维数非法
        BIZLOG(ErrorCode::kOpRankInvalid, "rmsNorm", "x and weight must be 1D");
        return Tensor();
    }

    int dim = x.shape[0];
    if (dim != weight.shape[0]) {
        // 错误码：算子输入维度不一致
        BIZLOG(ErrorCode::kOpDimMismatch, "rmsNorm",
               "x and weight must have the same dimension, got " +
                   x.shapeString() + " and " + weight.shapeString());
        return Tensor();
    }

    float ss = 0.0;
    for (int i = 0; i < dim; ++i) {
        ss += x[i] * x[i];
    }
    float scale = std::sqrt(ss / static_cast<float>(dim) + eps);

    Tensor result{{dim}, 0.0f};
    for (int i = 0; i < dim; ++i) {
        result[i] = x[i] * weight[i] / scale;
    }

    return result;
}

// 将相关性得分转换为注意力权重
Tensor softmax(const Tensor& x) {
    if (x.numDims() != 1) {
        // 错误码：算子输入维数非法
        BIZLOG(ErrorCode::kOpRankInvalid, "softmax",
               "x must be 1D, but got " + x.shapeString());
        return Tensor();
    }

    if (x.numElements() == 0) {
        // 错误码：算子输入为空
        BIZLOG(ErrorCode::kOpEmptyInput, "softmax");
        return Tensor();
    }

    // 获取最大值
    int dim = x.shape[0];
    float max = x[0];
    for (int i = 1; i < dim; ++i) {
        max = std::max(max, x[i]);
    }

    // 计算指数，同时计算概率因子
    float sum = 0.0f;
    Tensor result{{dim}, 0.0f};
    for (int i = 0; i < dim; ++i) {
        result[i] = std::exp(x[i] - max);
        sum += result[i];
    }

    // 得分转换为概率
    for (int i = 0; i < dim; ++i) {
        result[i] /= sum;
    }

    return result;
}

Tensor silu(const Tensor& x) {
    int dim = x.shape[0];
    Tensor result{{x.shape}, 0.0f};
    for (int i = 0; i < dim; ++i) {
        float val = x[i];
        result[i] = val / (1.0 + std::exp(-val));
    }
    return result;
}

Tensor elementwiseMul(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        // 错误码：算子输入形状不一致
        BIZLOG(ErrorCode::kOpShapeMismatch, "elementwiseMul", a.shapeString(),
               b.shapeString());
        return Tensor();
    }

    Tensor result{{a.shape}, 0.0f};
    for (int i = 0; i < a.data.size(); ++i) {
        result[i] = a[i] * b[i];
    }

    return result;
}

Tensor swiGlu(const Tensor& gate, const Tensor& up) {
    if (gate.shape != up.shape) {
        // 错误码：算子输入形状不一致
        BIZLOG(ErrorCode::kOpShapeMismatch, "swiGlu", gate.shapeString(),
               up.shapeString());
        return Tensor();
    }
    return elementwiseMul(silu(gate), up);
}

// 相邻维度特征rope
static void normalRope(int pos, float theta, Tensor& x) {
    int n_heads = x.shape[0];
    int head_dim = x.shape[1];
    for (int head = 0; head < n_heads; ++head) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq =
                1.0f / std::pow(theta, static_cast<float>(i) /
                                           static_cast<float>(head_dim));
            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);
            float x0 = x.data[head * head_dim + i];
            float x1 = x.data[head * head_dim + i + 1];
            x.data[head * head_dim + i] = x0 * cos_val - x1 * sin_val;
            x.data[head * head_dim + i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

//
static void neoxRope(int pos, float theta, Tensor& x) {
    int n_heads = x.shape[0];
    int head_dim = x.shape[1];
    int half_dim = head_dim / 2;
    for (int head = 0; head < n_heads; ++head) {
        int base = head * head_dim;
        for (int i = 0; i < half_dim; ++i) {
            float freq =
                1.0f / std::pow(theta, static_cast<float>(2 * i) /
                                           static_cast<float>(head_dim));
            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);
            float x0 = x.data[base + i];
            float x1 = x.data[base + half_dim + i];
            x.data[base + i] = x0 * cos_val - x1 * sin_val;
            x.data[base + half_dim + i] = x0 * sin_val + x1 * cos_val;
        }
    }
}

bool rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type) {
    // q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
    if (q.numDims() != 2 || k.numDims() != 2) {
        BIZLOG(ErrorCode::kRopeError, "Rope: expected 2D tensors");
        return false;
    }
    if (pos < 0) {
        BIZLOG(ErrorCode::kRopeError, "Rope: position must be non-negative");
        return false;
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
        BIZLOG(ErrorCode::kRopeError,
               "Rope: theta must be finite and positive");
        return false;
    }

    int n_heads = q.shape[0];
    int head_dim = q.shape[1];
    int n_kv_heads = k.shape[0];
    int k_head_dim = k.shape[1];
    if (head_dim != k_head_dim) {
        BIZLOG(ErrorCode::kRopeError, "Rope: q and k head_dim mismatch " +
                                          q.shapeString() + " vs " +
                                          k.shapeString());
        return false;
    }
    if (head_dim <= 0 || head_dim % 2 != 0) {
        BIZLOG(ErrorCode::kRopeError,
               "Rope: head_dim must be positive and even, got " +
                   std::to_string(head_dim));
        return false;
    }

    if (rope_type == RopeType::kNeoX) {
        neoxRope(pos, theta, q);
        neoxRope(pos, theta, k);
    } else {
        normalRope(pos, theta, q);
        normalRope(pos, theta, k);
    }
    return true;
}

// 获取概率分布中最大值的索引
int argMax(const Tensor& x) {
    if (x.size() == 0) {
        // 错误码：算子输入为空
        BIZLOG(ErrorCode::kOpEmptyInput, "argMax");
        return -1;
    }
    int best = 0;
    float best_val = x.data[0];
    for (size_t i = 1; i < x.data.size(); ++i) {
        if (x.data[i] > best_val) {
            best_val = x.data[i];
            best = static_cast<int>(i);
        }
    }
    return best;
}

}  // namespace mini_llama
