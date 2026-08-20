// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/matmul_dispatch.h"

#include "errlog/bizlog.h"          // 错误码日志：BIZLOG / ErrorCode
#include "mini_llama/threadpool.h"  // 线程池：ThreadPool

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

// ---------------------------------------------------------------------------
// Threaded implementations
// ---------------------------------------------------------------------------

static Tensor matmulThreaded(const Tensor& a, const Tensor& b) {
    int m = a.shape[0];
    int inner_dim = a.shape[1];
    int n = b.shape[1];
    Tensor c({m, n}, 0.0f);
    ThreadPool::submitTask(m, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < inner_dim; ++k) {
                    sum += a.data[i * inner_dim + k] * b.data[k * n + j];
                }
                c.data[i * n + j] = sum;
            }
        }
    });

    return c;
}

static Tensor linearThreaded(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    bool is_1d = (x.numDims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                  0.0f);
    ThreadPool::submitTask(out_features, [&](int begin, int end) {
        for (int j = begin; j < end; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                sum += x.data[k] * weight.data[j * in_features + k];
            }
            result.data[j] = sum;
        }
    });
    return result;
}

// ---------------------------------------------------------------------------
// SIMD implementations
// ---------------------------------------------------------------------------

static float dotSimd(const float* a, const float* b, int n) {
#if defined(MINI_LLAMA_USE_AVX2)
    return dotSimdAvx2(a, b, n);
#elif defined(MINI_LLAMA_USE_NEON)
    return dotSimdNeon(a, b, n);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

#ifdef MINI_LLAMA_USE_NEON
static float dotSimdNeon(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }
    float32x2_t r = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    r = vpadd_f32(r, r);
    float result = vget_lane_f32(r, 0);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

#ifdef MINI_LLAMA_USE_AVX2
static float dotSimdAvx2(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    // Horizontal reduce
    __m128 sum_lo = _mm256_castps256_ps128(sum);
    __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
    sum_lo = _mm_add_ps(sum_lo, sum_hi);
    sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
    sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
    float result = _mm_cvtss_f32(sum_lo);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif

static Tensor linearSimd(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    bool is_1d = (x.numDims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                  0.0f);
    for (int j = 0; j < out_features; ++j) {
        result.data[j] = dotSimd(
            x.data.data(), weight.data.data() + j * in_features, in_features);
    }
    return result;
}

static Tensor linearThreadedSimd(const Tensor& x, const Tensor& weight) {
    int in_features = weight.shape[1];
    int out_features = weight.shape[0];
    bool is_1d = (x.numDims() == 1);

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{1, out_features},
                  0.0f);
    ThreadPool::submitTask(out_features, [&](int begin, int end) {
        for (int j = begin; j < end; ++j) {
            result.data[j] =
                dotSimd(x.data.data(), weight.data.data() + j * in_features,
                        in_features);
        }
    });
    return result;
}

// 根据MatmulMode选择合适的实现
MatmulMode defaultMatmulMode() { return MatmulMode::kThreadedSimd; }

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
        case MatmulMode::kSimd:
        case MatmulMode::kThreadedSimd:
            return matmulThreaded(a, b);
    }

    return matmulThreaded(a, b);
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
            return linearThreaded(x, weight);
        case MatmulMode::kSimd:
            return linearSimd(x, weight);
        case MatmulMode::kThreadedSimd:
            return linearThreadedSimd(x, weight);
    }

    return linearNaive(x, weight);
}

}  // namespace mini_llama
