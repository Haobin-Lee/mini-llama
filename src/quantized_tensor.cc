// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/quantized_tensor.h"

#include "errlog/bizlog.h"

namespace mini_llama {

using errlog::ErrorCode;

Tensor toTensor(const QuantizedTensor& q) {
    if (q.type != QuantType::kF32) {
        BIZLOG(ErrorCode::kQuantTypeMismatch,
               "ToTensor: expected F32 QuantizedTensor, got quantized type");
        return Tensor(q.shape, 0.0f);
    }
    Tensor t(q.shape, 0.0f);
    if (q.f32_data.size() == t.data.size()) {
        t.data = q.f32_data;
    }
    return t;
}

QuantizedTensor toQuantizedTensor(const Tensor& t) {
    QuantizedTensor q;
    q.type = QuantType::kF32;
    q.shape = t.shape;
    q.f32_data = t.data;
    return q;
}

}  // namespace mini_llama
