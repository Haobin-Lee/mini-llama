// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/matmul_dispatch.h"

#include "errlog/bizlog.h"  // 错误码日志：BIZLOG / ErrorCode

namespace mini_llama {

using errlog::ErrorCode;

static Tensor matmulNaive(const Tensor& a, const Tensor& b) {
    int row = a.shape[0];
    int col = b.shape[1];
    int in_features = a.shape[1];

    Tensor result({row, col}, 0.0f);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                sum += a.data[i * in_features + k] * b.data[k * col + j];
            }
            result.data[i * col + j] = sum;
        }
    }

    return result;
}

static Tensor linearNaive(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];

    bool is_1d = (x.shape.size() == 1);
    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                  0.0f);

    for (int i = 0; i < out_features; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < in_features; ++j) {
            sum += x.data[j] * weight.data[i * in_features + j];
        }
        result.data[i] = sum;
    }

    return result;
}

// 根据MatmulMode选择合适的实现
// TODO: 当前只有kNaive普通实现， 后续可以 多线程 + SIMD优化
MatmulMode defaultMatmulMode() { return MatmulMode::kNaive; }

Tensor matmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        // 错误码：算子输入维数非法
        BIZLOG(ErrorCode::kOpRankInvalid, "matmulDispatch",
               "a and b must be 2D tensors");
        return Tensor();
    }

    if (a.shape[1] != b.shape[0]) {
        // 错误码：矩阵乘形状不兼容
        BIZLOG(ErrorCode::kMatmulShapeIncompatible, "matmulDispatch",
               a.shapeString(), b.shapeString());
        return Tensor();
    }

    switch (mode) {
        case MatmulMode::kNaive:
            return matmulNaive(a, b);
        case MatmulMode::kThreaded:
            // TODO: 多线程实现
            break;
        case MatmulMode::kSimd:
            // TODO: SIMD实现
            break;
        case MatmulMode::kThreadedSimd:
            // TODO: 多线程 + SIMD实现
            // matmul是跨列访问的，缓存不友好，SIMD会比较复杂，所以这里使用多线程
            break;
    }

    return matmulNaive(a, b);
}

Tensor linearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
    if (weight.shape.size() != 2) {
        // 错误码：算子输入维数非法
        BIZLOG(ErrorCode::kOpRankInvalid, "linearDispatch",
               "weight must be 2D tensor");
        return Tensor();
    }

    int in_features = 0;
    if (x.numDims() == 1) {
        in_features = x.shape[0];
    } else if (x.numDims() == 2) {
        in_features = x.shape[1];
    } else {
        // 错误码：算子输入维数非法
        BIZLOG(ErrorCode::kOpRankInvalid, "linearDispatch",
               "x must be 1D or 2D, but got " + x.shapeString());
        return Tensor();
    }

    if (weight.shape[1] != in_features) {
        // 错误码：矩阵乘形状不兼容
        BIZLOG(ErrorCode::kMatmulShapeIncompatible, "linearDispatch",
               weight.shapeString(), x.shapeString());
        return Tensor();
    }

    switch (mode) {
        case MatmulMode::kNaive:
            return linearNaive(x, weight);
            break;
        case MatmulMode::kThreaded:
            // TODO: 多线程实现
            break;
        case MatmulMode::kSimd:
            // TODO: SIMD实现
            break;
        case MatmulMode::kThreadedSimd:
            // TODO: 多线程 + SIMD实现
            break;
    }

    return linearNaive(x, weight);
}

}  // namespace mini_llama
