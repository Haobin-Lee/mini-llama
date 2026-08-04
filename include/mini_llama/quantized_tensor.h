// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_
#define INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "errlog/bizlog.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

using errlog::ErrorCode;

// ---------------------------------------------------------------------------
// Quantization types
// ---------------------------------------------------------------------------

enum class QuantType {
    kF32,
    kQ80,
    kQ40,
    kQ41,
};

// ---------------------------------------------------------------------------
// Q8_0 block layout (ggml / llama.cpp compatible)
// Each block covers 32 elements.
// ---------------------------------------------------------------------------
constexpr int kQ80BlockSize = 32;

struct BlockQ80 {
    uint16_t d;     // FP16 delta (scale)
    int8_t qs[32];  // quantized values
};

static_assert(sizeof(BlockQ80) == sizeof(uint16_t) + kQ80BlockSize,
              "wrong Q8_0 block size");

// ---------------------------------------------------------------------------
// Q4_0 block layout (ggml / llama.cpp compatible)
// Each block covers 32 elements, packed into 16 bytes.
// ---------------------------------------------------------------------------
constexpr int kQ40BlockSize = 32;

struct BlockQ40 {
    uint16_t d;      // FP16 delta (scale)
    uint8_t qs[16];  // 32 x 4-bit values, packed (2 per byte)
};

static_assert(sizeof(BlockQ40) == sizeof(uint16_t) + kQ40BlockSize / 2,
              "wrong Q4_0 block size");

// ---------------------------------------------------------------------------
// Q4_1 block layout (ggml / llama.cpp compatible)
// Each block covers 32 elements, packed into 16 bytes.
// Value formula: x = d * q + m, q in [0, 15].
// ---------------------------------------------------------------------------
constexpr int kQ41BlockSize = 32;

struct BlockQ41 {
    uint16_t d;      // FP16 delta (scale)
    uint16_t m;      // FP16 minimum/offset
    uint8_t qs[16];  // 32 x 4-bit values, packed (2 per byte)
};

static_assert(sizeof(BlockQ41) == sizeof(uint16_t) * 2 + kQ41BlockSize / 2,
              "wrong Q4_1 block size");

// ---------------------------------------------------------------------------
// QuantizedTensor: tagged struct holding quantized or F32 data.
//
// For teaching clarity, uses a tagged union-like layout rather than
// std::variant.  Only one of {f32_data, q8_0_data, q4_0_data, q4_1_data} is
// valid depending on `type`.
// ---------------------------------------------------------------------------
struct QuantizedTensor {
    QuantType type = QuantType::kF32;
    std::vector<int> shape;

    // Valid when type == F32
    std::vector<float> f32_data;

    // Valid when type == Q8_0
    std::vector<BlockQ80> q8_0_data;

    // Valid when type == Q4_0
    std::vector<BlockQ40> q4_0_data;

    // Valid when type == Q4_1
    std::vector<BlockQ41> q4_1_data;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    size_t num_elements() const {
        size_t n = 1;
        for (int d : shape) {
            n *= static_cast<size_t>(d);
        }
        return n;
    }

    std::string shapeStringShort() const {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (i + 1 < shape.size()) {
                s += ", ";
            }
        }
        s += "]";
        return s;
    }

    bool isSameShape(const std::vector<int>& expected,
                     const char* caller) const {
        if (shape != expected) {
            BIZLOG(
                ErrorCode::kTensorShapeMismatch, std::string(caller),
                QuantizedTensor{QuantType::kF32, expected}.shapeStringShort(),
                shapeStringShort());
            return false;
        }

        return true;
    }

    int num_dims() const { return static_cast<int>(shape.size()); }
};

// Convert a QuantizedTensor (type==F32) to a plain Tensor.
Tensor toTensor(const QuantizedTensor& q);

// Convert a plain Tensor to a QuantizedTensor (type==F32).
QuantizedTensor toQuantizedTensor(const Tensor& t);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_
