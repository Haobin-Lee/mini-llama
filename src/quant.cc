// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "errlog/bizlog.h"
#include "mini_llama/ops.h"
#include "mini_llama/threadpool.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define MINI_LLAMA_USE_AVX2 1
#endif

// TODO: arm neon 量化头文件

namespace mini_llama {

namespace {

    uint16_t floatToFp16(float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));

        uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exponent =
            static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
        uint32_t mantissa = bits & 0x7fffffu;

        if (exponent <= 0) {
            if (exponent < -10) {
                return static_cast<uint16_t>(sign);
            }
            mantissa |= 0x800000u;
            uint32_t shifted = mantissa >> (1 - exponent);
            if ((shifted & 0x00001000u) != 0) {
                shifted += 0x00002000u;
            }
            return static_cast<uint16_t>(sign | (shifted >> 13));
        }

        if (exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7c00u);
        }

        if ((mantissa & 0x00001000u) != 0) {
            mantissa += 0x00002000u;
            if ((mantissa & 0x00800000u) != 0) {
                mantissa = 0;
                ++exponent;
                if (exponent >= 31) {
                    return static_cast<uint16_t>(sign | 0x7c00u);
                }
            }
        }

        return static_cast<uint16_t>(
            sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
    }

    float fp16ToFloat(uint16_t value) {
        uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
        uint32_t exponent = (value >> 10) & 0x1fu;
        uint32_t mantissa = value & 0x03ffu;
        uint32_t bits = 0;

        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                exponent = 1;
                while ((mantissa & 0x0400u) == 0) {
                    mantissa <<= 1;
                    --exponent;
                }
                mantissa &= 0x03ffu;
                uint32_t exp32 = exponent + (127 - 15);
                bits = sign | (exp32 << 23) | (mantissa << 13);
            }
        } else if (exponent == 31) {
            bits = sign | 0x7f800000u | (mantissa << 13);
        } else {
            uint32_t exp32 = exponent + (127 - 15);
            bits = sign | (exp32 << 23) | (mantissa << 13);
        }

        float out = 0.0f;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    size_t checkedNumel(const std::vector<int>& shape, const char* caller) {
        size_t total = 1;
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            int dim = shape[axis];
            if (dim <= 0) {
                BIZLOG(ErrorCode::kQuantError,
                       std::string(caller) + ": dimension at axis " +
                           std::to_string(axis) + " must be positive, got " +
                           std::to_string(dim));
                return 0;
            }
            size_t dim_size = static_cast<size_t>(dim);
            if (total > std::numeric_limits<size_t>::max() / dim_size) {
                BIZLOG(ErrorCode::kQuantError,
                       std::string(caller) + ": shape element count overflow");
                return 0;
            }
            total *= dim_size;
        }
        return total;
    }

}  // namespace

// ---------------------------------------------------------------------------
// Q8_0 quantize / dequantize
// ---------------------------------------------------------------------------
std::vector<BlockQ80> quantizeToQ80(const Tensor& src) {
    if (src.size() == 0) {
        return {};
    }

    int row_size =
        src.numDims() >= 2 ? src.shape.back() : static_cast<int>(src.size());
    int n_rows =
        src.numDims() >= 2 ? static_cast<int>(src.size()) / row_size : 1;
    int row_blocks = (row_size + kQ80BlockSize - 1) / kQ80BlockSize;
    size_t total_blocks = static_cast<size_t>(n_rows) * row_blocks;

    std::vector<BlockQ80> blocks;
    blocks.reserve(total_blocks);

    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * kQ80BlockSize;
            int k_end = std::min(base + kQ80BlockSize, row_offset + row_size);
            int block_len = k_end - base;

            float max_abs = 0.0f;
            for (int k = base; k < k_end; ++k) {
                float abs_val = std::abs(src.data[k]);
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                }
            }

            BlockQ80 block;
            std::memset(&block, 0, sizeof(block));
            if (max_abs > 0.0f) {
                float d = max_abs / 127.0f;
                block.d = floatToFp16(d);
                float stored_d = fp16ToFloat(block.d);
                float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;
                for (int i = 0; i < block_len; ++i) {
                    float q = src.data[base + i] * id;
                    int qi = static_cast<int>(std::round(q));
                    if (qi > 127) {
                        qi = 127;
                    } else if (qi < -127) {
                        qi = -127;
                    }
                    block.qs[i] = static_cast<int8_t>(qi);
                }
            } else {
                block.d = 0;
            }
            // Pad remainder with 0
            for (int i = block_len; i < kQ80BlockSize; ++i) {
                block.qs[i] = 0;
            }

            blocks.push_back(block);
        }
    }

    return blocks;
}

Tensor dequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape) {
    size_t total = checkedNumel(shape, "dequantizeFromQ80");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int row_blocks = (row_size + kQ80BlockSize - 1) / kQ80BlockSize;
    size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
    if (blocks.size() != expected_blocks) {
        BIZLOG(ErrorCode::kQuantError,
               std::string("dequantizeFromQ80") + ": block count mismatch: " +
                   std::to_string(expected_blocks) + ", got " +
                   std::to_string(blocks.size()));
    }

    Tensor dst(shape, 0.0f);
    size_t block_idx = 0;
    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * kQ80BlockSize;
            int k_end = std::min(base + kQ80BlockSize, row_offset + row_size);
            const BlockQ80& block = blocks[block_idx++];
            float d = fp16ToFloat(block.d);
            for (int k = base; k < k_end; ++k) {
                dst.data[k] = d * static_cast<float>(block.qs[k - base]);
            }
        }
    }

    return dst;
}

// ---------------------------------------------------------------------------
// Q4_0 quantize / dequantize
// ---------------------------------------------------------------------------
std::vector<BlockQ40> quantizeToQ40(const Tensor& src) {
    if (src.size() == 0) {
        return {};
    }

    int row_size =
        src.numDims() >= 2 ? src.shape.back() : static_cast<int>(src.size());
    int n_rows =
        src.numDims() >= 2 ? static_cast<int>(src.size()) / row_size : 1;
    int row_blocks = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
    size_t total_blocks = static_cast<size_t>(n_rows) * row_blocks;

    std::vector<BlockQ40> blocks;
    blocks.reserve(total_blocks);

    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * kQ40BlockSize;
            int k_end = std::min(base + kQ40BlockSize, row_offset + row_size);
            int block_len = k_end - base;

            float max_abs = 0.0f;
            for (int k = base; k < k_end; ++k) {
                float abs_val = std::abs(src.data[k]);
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                }
            }

            BlockQ40 block;
            std::memset(&block, 0, sizeof(block));
            if (max_abs > 0.0f) {
                float d =
                    max_abs / 7.0f;  // max representable is 7 (q=15 -> 15-8=7)
                block.d = floatToFp16(d);
                float stored_d = fp16ToFloat(block.d);
                float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;

                for (int j = 0; j < kQ40BlockSize / 2; ++j) {
                    float x0 = 0.0f, x1 = 0.0f;
                    int idx0 = base + j;
                    int idx1 = base + j + kQ40BlockSize / 2;
                    if (idx0 < k_end) {
                        x0 = src.data[idx0] * id;
                    }
                    if (idx1 < k_end) {
                        x1 = src.data[idx1] * id;
                    }
                    int qi0 = static_cast<int>(std::round(x0 + 8.0f));
                    int qi1 = static_cast<int>(std::round(x1 + 8.0f));
                    if (qi0 < 0) {
                        qi0 = 0;
                    }
                    if (qi0 > 15) {
                        qi0 = 15;
                    }
                    if (qi1 < 0) {
                        qi1 = 0;
                    }
                    if (qi1 > 15) {
                        qi1 = 15;
                    }
                    block.qs[j] = static_cast<uint8_t>(qi0 | (qi1 << 4));
                }
            } else {
                block.d = 0;
            }

            blocks.push_back(block);
        }
    }

    return blocks;
}

Tensor dequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape) {
    size_t total = checkedNumel(shape, "DequantizeFromQ40");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int row_blocks = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
    size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
    if (blocks.size() != expected_blocks) {
        BIZLOG(ErrorCode::kQuantError,
               std::string("dequantizeFromQ40") + ": block count mismatch: " +
                   std::to_string(expected_blocks) + ", got " +
                   std::to_string(blocks.size()));
    }

    Tensor dst(shape, 0.0f);
    size_t block_idx = 0;
    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * kQ40BlockSize;
            int k_end = std::min(base + kQ40BlockSize, row_offset + row_size);
            const BlockQ40& block = blocks[block_idx++];
            float d = fp16ToFloat(block.d);

            for (int j = 0; j < kQ40BlockSize / 2; ++j) {
                int idx0 = base + j;
                int idx1 = base + j + kQ40BlockSize / 2;
                int q0 = static_cast<int>(block.qs[j] & 0x0F) - 8;
                int q1 = static_cast<int>(block.qs[j] >> 4) - 8;
                if (idx0 < k_end) {
                    dst.data[idx0] = d * static_cast<float>(q0);
                }
                if (idx1 < k_end) {
                    dst.data[idx1] = d * static_cast<float>(q1);
                }
            }
        }
    }

    return dst;
}

// ---------------------------------------------------------------------------
// Q4_1 dequantize
// ---------------------------------------------------------------------------
Tensor dequantizeFromQ41(const std::vector<BlockQ41>& blocks,
                         const std::vector<int>& shape) {
    size_t total = checkedNumel(shape, "dequantizeFromQ41");

    int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
    int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
    int row_blocks = (row_size + kQ41BlockSize - 1) / kQ41BlockSize;
    size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
    if (blocks.size() != expected_blocks) {
        BIZLOG(ErrorCode::kQuantError,
               std::string("dequantizeFromQ41") + ": block count mismatch: " +
                   std::to_string(expected_blocks) + ", got " +
                   std::to_string(blocks.size()));
    }

    Tensor dst(shape, 0.0f);
    size_t block_idx = 0;
    for (int row = 0; row < n_rows; ++row) {
        int row_offset = row * row_size;
        for (int rb = 0; rb < row_blocks; ++rb) {
            int base = row_offset + rb * kQ41BlockSize;
            int k_end = std::min(base + kQ41BlockSize, row_offset + row_size);
            const BlockQ41& block = blocks[block_idx++];
            float d = fp16ToFloat(block.d);
            float m = fp16ToFloat(block.m);

            for (int j = 0; j < kQ41BlockSize / 2; ++j) {
                int idx0 = base + j;
                int idx1 = base + j + kQ41BlockSize / 2;
                int q0 = static_cast<int>(block.qs[j] & 0x0F);
                int q1 = static_cast<int>(block.qs[j] >> 4);
                if (idx0 < k_end) {
                    dst.data[idx0] = d * static_cast<float>(q0) + m;
                }
                if (idx1 < k_end) {
                    dst.data[idx1] = d * static_cast<float>(q1) + m;
                }
            }
        }
    }

    return dst;
}

// ---------------------------------------------------------------------------
// True quantized Linear (block-level on-the-fly)
// ---------------------------------------------------------------------------

// Shared helper: Linear with quantized 2D weight [out_features, in_features]
// and 1D input [in_features] or 2D input [batch, in_features].
// result rank matches input rank.
template <typename BlockType, int kBlockSize>
static Tensor linearQuantizedImpl(const Tensor& x,
                                  const std::vector<BlockType>& blocks,
                                  const std::vector<int>& weight_shape,
                                  float (*dequant_fn)(const BlockType&, int)) {
    if (weight_shape.size() != 2) {
        BIZLOG(ErrorCode::kQuantError,
               "linear_quantized: expected 2D weight shape");
        return Tensor();
    }

    int in_features;
    int rows = 1;
    bool is_1d = false;
    if (x.numDims() == 1) {
        in_features = x.shape[0];
        is_1d = true;
    } else if (x.numDims() == 2) {
        rows = x.shape[0];
        in_features = x.shape[1];
    } else {
        BIZLOG(ErrorCode::kQuantError,
               "linear_quantized: expected x shape [in_features] or [batch, "
               "in_features], got " +
                   x.shapeString());
        return Tensor();
    }

    int out_features = weight_shape[0];
    if (weight_shape[1] != in_features) {
        BIZLOG(ErrorCode::kQuantError,
               "linear_quantized: dimension mismatch x=%s weight=%s",
               x.shapeString(),
               QuantizedTensor{QuantType::kF32, weight_shape}.shapeString());
        return Tensor();
    }

    int n_blocks_per_row = (in_features + kBlockSize - 1) / kBlockSize;
    size_t expected_blocks =
        static_cast<size_t>(out_features) * n_blocks_per_row;
    if (blocks.size() != expected_blocks) {
        BIZLOG(ErrorCode::kQuantError,
               std::string("linear_quantized") + ": block count mismatch: " +
                   std::to_string(expected_blocks) + ", got " +
                   std::to_string(blocks.size()));
        return Tensor();
    }

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{rows, out_features},
                  0.0f);

    ThreadPool::submitTask(rows * out_features, [&](int begin, int end) {
        for (int flat_index = begin; flat_index < end; ++flat_index) {
            int row = flat_index / out_features;
            int j = flat_index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;
            float sum = 0.0f;
            int block_base = j * n_blocks_per_row;
            for (int b = 0; b < n_blocks_per_row; ++b) {
                const BlockType& block = blocks[block_base + b];
                int base_k = b * kBlockSize;
                int k_end = std::min(base_k + kBlockSize, in_features);
                for (int k = base_k; k < k_end; ++k) {
                    float w = dequant_fn(block, k - base_k);
                    sum += w * x_row[k];
                }
            }
            result.data[static_cast<size_t>(row) * out_features + j] = sum;
        }
    });

    return result;
}

static float dequantQ80(const BlockQ80& block, int idx) {
    float d = fp16ToFloat(block.d);
    return d * static_cast<float>(block.qs[idx]);
}

static float dequantQ40(const BlockQ40& block, int idx) {
    float d = fp16ToFloat(block.d);
    int q;
    if (idx < 16) {
        q = static_cast<int>(block.qs[idx] & 0x0F) - 8;
    } else {
        q = static_cast<int>(block.qs[idx - 16] >> 4) - 8;
    }
    return d * static_cast<float>(q);
}

static float dequantQ41(const BlockQ41& block, int idx) {
    float d = fp16ToFloat(block.d);
    float m = fp16ToFloat(block.m);
    int q;
    if (idx < 16) {
        q = static_cast<int>(block.qs[idx] & 0x0F);
    } else {
        q = static_cast<int>(block.qs[idx - 16] >> 4);
    }
    return d * static_cast<float>(q) + m;
}

#ifdef MINI_LLAMA_USE_AVX2
// AVX2-optimized Q8_0 Linear: process 8 quantized values per SIMD iteration.
Tensor linearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
    if (weight_shape.size() != 2) {
        BIZLOG(ErrorCode::kQuantError, "LinearQ80: expected 2D weight shape");
        return Tensor();
    }

    int in_features;
    int rows = 1;
    bool is_1d = false;
    if (x.numDims() == 1) {
        in_features = x.shape[0];
        is_1d = true;
    } else if (x.numDims() == 2) {
        rows = x.shape[0];
        in_features = x.shape[1];
    } else {
        BIZLOG(ErrorCode::kQuantError,
               "LinearQ80: expected x shape [in_features] or [batch, "
               "in_features], "
               "got " +
                   x.shapeString());
        return Tensor();
    }

    int out_features = weight_shape[0];
    if (weight_shape[1] != in_features) {
        BIZLOG(
            ErrorCode::kQuantError,
            "LinearQ80: dimension mismatch x=" + x.shapeString() + " weight=" +
                QuantizedTensor{QuantType::kF32, weight_shape}.shapeString());
        return Tensor();
    }

    int n_blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;
    size_t expected_blocks =
        static_cast<size_t>(out_features) * n_blocks_per_row;
    if (weight.size() != expected_blocks) {
        BIZLOG(ErrorCode::kQuantError, "LinearQ80: block count mismatch: " +
                                           std::to_string(expected_blocks) +
                                           ", got " +
                                           std::to_string(weight.size()));
        return Tensor();
    }

    Tensor result(is_1d ? std::vector<int>{out_features}
                        : std::vector<int>{rows, out_features},
                  0.0f);

    ThreadPool::submitTask(rows * out_features, [&](int begin, int end) {
        for (int flat_index = begin; flat_index < end; ++flat_index) {
            int row = flat_index / out_features;
            int j = flat_index % out_features;
            const float* x_row =
                x.data.data() + static_cast<size_t>(row) * in_features;
            int block_base = j * n_blocks_per_row;
            __m256 sum_vec = _mm256_setzero_ps();
            float sum_scalar = 0.0f;

            for (int b = 0; b < n_blocks_per_row; ++b) {
                const BlockQ80& block = weight[block_base + b];
                float d = fp16ToFloat(block.d);
                int base_k = b * kQ80BlockSize;
                int k_end = std::min(base_k + kQ80BlockSize, in_features);
                // Process 8 elements at a time (1x __m256)
                int k = base_k;
                for (; k + 8 <= k_end; k += 8) {
                    // Load 8 int8 quantized values
                    __m128i q8 =
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(
                            block.qs + (k - base_k)));

                    // Sign-extend int8 to int16
                    __m256i q16 = _mm256_cvtepi8_epi16(q8);

                    // Sign-extend int16 to int32 (split into two 128-bit
                    // halves)
                    __m256i q32_lo =
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16));
                    __m256i q32_hi =
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1));

                    // Convert int32 to float
                    __m256 qf_lo = _mm256_cvtepi32_ps(q32_lo);
                    __m256 qf_hi = _mm256_cvtepi32_ps(q32_hi);

                    // Multiply by scale factor
                    __m256 scale = _mm256_set1_ps(d);
                    qf_lo = _mm256_mul_ps(qf_lo, scale);
                    qf_hi = _mm256_mul_ps(qf_hi, scale);

                    // Load input x
                    __m256 xf_lo = _mm256_loadu_ps(x_row + k);
                    __m256 xf_hi = _mm256_loadu_ps(x_row + k + 4);

                    // FMA accumulation
                    sum_vec = _mm256_fmadd_ps(qf_lo, xf_lo, sum_vec);
                    sum_vec = _mm256_fmadd_ps(qf_hi, xf_hi, sum_vec);
                }

                // Horizontal reduce the 8-lane accumulator
                __m128 sum_lo = _mm256_castps256_ps128(sum_vec);
                __m128 sum_hi = _mm256_extractf128_ps(sum_vec, 1);
                sum_lo = _mm_add_ps(sum_lo, sum_hi);
                sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
                sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
                sum_scalar += _mm_cvtss_f32(sum_lo);
                sum_vec = _mm256_setzero_ps();

                // Tail: remaining elements (< 8)
                for (; k < k_end; ++k) {
                    float w = d * static_cast<float>(block.qs[k - base_k]);
                    sum_scalar += w * x_row[k];
                }
            }

            result.data[static_cast<size_t>(row) * out_features + j] =
                sum_scalar;
        }
    });
    return result;
}
#else
Tensor linearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
    return linearQuantizedImpl<BlockQ80, kQ80BlockSize>(x, weight, weight_shape,
                                                        dequantQ80);
}
#endif

// TODO: NEON-optimized Q4_0 Linear: unpack 4-bit nibbles and process 16 at a
// time.

Tensor linearQ40(const Tensor& x, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape) {
    return linearQuantizedImpl<BlockQ40, kQ40BlockSize>(x, weight, weight_shape,
                                                        dequantQ40);
}

Tensor linearQ41(const Tensor& x, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape) {
    return linearQuantizedImpl<BlockQ41, kQ41BlockSize>(x, weight, weight_shape,
                                                        dequantQ41);
}

// ---------------------------------------------------------------------------
// Legacy pseudo-quantized Matmul (dequantizes to F32 then calls Matmul)
// ---------------------------------------------------------------------------
Tensor matmulQ80(const std::vector<BlockQ80>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape) {
    Tensor weight_f32 = dequantizeFromQ80(weight, weight_shape);
    return matmul(weight_f32, input);
}

Tensor matmulQ40(const std::vector<BlockQ40>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape) {
    Tensor weight_f32 = dequantizeFromQ40(weight, weight_shape);
    return matmul(weight_f32, input);
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------
float compareMatmulError(const Tensor& weight, const Tensor& input) {
    Tensor f32_result = matmul(weight, input);
    std::vector<BlockQ80> qweight = quantizeToQ80(weight);
    Tensor q8_result = matmulQ80(qweight, input, weight.shape);

    if (f32_result.shape != q8_result.shape) {
        BIZLOG(ErrorCode::kQuantError, "compareMatmulError: shape mismatch");
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < f32_result.size(); ++i) {
        float err = std::abs(f32_result.data[i] - q8_result.data[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

float compareQ40Error(const Tensor& weight, const Tensor& input) {
    Tensor f32_result = linear(input, weight);
    std::vector<BlockQ40> qweight = quantizeToQ40(weight);
    Tensor q4_result = linearQ40(input, qweight, weight.shape);

    if (f32_result.shape != q4_result.shape) {
        BIZLOG(ErrorCode::kQuantError, "compareQ40Error: shape mismatch");
        return 0.0f;
    }

    float max_err = 0.0f;
    for (size_t i = 0; i < f32_result.size(); ++i) {
        float err = std::abs(f32_result.data[i] - q4_result.data[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

}  // namespace mini_llama
