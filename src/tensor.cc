// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/tensor.h"

#include <limits>
#include <numeric>

#include "errlog/bizlog.h"  // 错误码日志：BIZLOG / ErrorCode

namespace mini_llama {

using errlog::ErrorCode;

namespace {
    const char* callerName(const char* caller) {
        return caller == nullptr ? "Tensor" : caller;
    }

    std::string shapeToString(const std::vector<int>& shape) {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); i++) {
            s += std::to_string(shape[i]);
            if (i < shape.size() - 1) {
                s += ", ";
            }
        }
        s += "]";
        return s;
    }

    size_t checkedNumel(const std::vector<int>& shape, const char* caller) {
        size_t total = 1;
        if (shape.empty()) {
            // 错误码：张量形状为空
            BIZLOG(ErrorCode::kTensorShapeEmpty,
                   std::string("caller= ") + callerName(caller));
            return 0;
        }

        for (int axis = 0; axis < shape.size(); ++axis) {
            int dim = shape[axis];
            if (dim <= 0) {
                // 错误码：维度非正
                BIZLOG(ErrorCode::kTensorDimNotPositive, callerName(caller),
                       axis, dim, shapeToString(shape));
                return 0;
            }
            if (total > std::numeric_limits<size_t>::max() / dim) {
                // 错误码：元素数量溢出（critical + [ALERT]）
                BIZLOG(ErrorCode::kTensorNumelOverflow, callerName(caller),
                       shapeToString(shape));
                return 0;
            }

            total *= dim;
        }

        return total;
    }

    void checkRank(const Tensor& t, int expected_rank, const char* caller) {
        if (t.numDims() != expected_rank) {
            // 错误码：维数不匹配
            BIZLOG(ErrorCode::kTensorRankMismatch, callerName(caller),
                   expected_rank, t.shapeString());
        }
    }

    void checkAxisIndex(const Tensor& t, int axis, int flat_index,
                        const char* caller) {
        const int dim = t.shape[axis];
        if (flat_index < 0 || flat_index >= dim) {
            // 错误码：下标越界
            BIZLOG(ErrorCode::kTensorIndexOutOfRange, callerName(caller),
                   flat_index, axis, dim, t.shapeString());
        }
    }
}  // namespace

Tensor::Tensor(const std::vector<int>& input_shape, float fill)
    : shape(input_shape) {
    const size_t total = checkedNumel(input_shape, "Tensor Constructor");
    data.resize(total, fill);
}

size_t Tensor::flatIndex(const std::vector<int>& indices) const {
    if (indices.size() != shape.size()) {
        // 错误码：索引个数与维度不一致
        BIZLOG(ErrorCode::kTensorIndexCountMismatch, shapeString(),
               shape.size(), indices.size());
        return 0;
    }

    for (size_t axis = 0; axis < indices.size(); ++axis) {
        checkAxisIndex(*this, axis, indices[axis], "Tensor::FlatIndex");
    }

    size_t flat_index = 0;
    size_t stride = 1;
    for (int axis = static_cast<int>(shape.size()) - 1; axis > 0; --axis) {
        flat_index += static_cast<size_t>(indices[axis]) * stride;
        stride *= static_cast<size_t>(shape[axis]);
    }

    return flat_index;
}

float& Tensor::at(const std::vector<int>& indices) {
    return data[flatIndex(indices)];
}

float Tensor::at(const std::vector<int>& indices) const {
    return data[flatIndex(indices)];
}

float& Tensor::at1(int i) {
    checkRank(*this, 1, "Tensor::at1");
    checkAxisIndex(*this, 0, i, "Tensor::at1");
    return data[i];
}
float Tensor::at1(int i) const {
    checkRank(*this, 1, "Tensor::at1(const)");
    checkAxisIndex(*this, 0, i, "Tensor::at1(const)");
    return data[i];
}
float& Tensor::at2(int i, int j) {
    checkRank(*this, 2, "Tensor::at2");
    checkAxisIndex(*this, 0, i, "Tensor::at2");
    checkAxisIndex(*this, 1, j, "Tensor::at2");
    return data[i * shape[1] + j];
}
float Tensor::at2(int i, int j) const {
    checkRank(*this, 2, "Tensor::at2(const)");
    checkAxisIndex(*this, 0, i, "Tensor::at2(const)");
    checkAxisIndex(*this, 1, j, "Tensor::at2(const)");
    return data[i * shape[1] + j];
}
float& Tensor::at3(int i, int j, int k) {
    checkRank(*this, 3, "Tensor::at3");
    checkAxisIndex(*this, 0, i, "Tensor::at3");
    checkAxisIndex(*this, 1, j, "Tensor::at3");
    checkAxisIndex(*this, 2, k, "Tensor::at3");
    return data[(i * shape[1] + j) * shape[2] + k];
}
float Tensor::at3(int i, int j, int k) const {
    checkRank(*this, 3, "Tensor::at3(const)");
    checkAxisIndex(*this, 0, i, "Tensor::at3(const)");
    checkAxisIndex(*this, 1, j, "Tensor::at3(const)");
    checkAxisIndex(*this, 2, k, "Tensor::at3(const)");
    return data[(i * shape[1] + j) * shape[2] + k];
}
float& Tensor::at4(int i, int j, int k, int l) {
    checkRank(*this, 4, "Tensor::at4");
    checkAxisIndex(*this, 0, i, "Tensor::at4");
    checkAxisIndex(*this, 1, j, "Tensor::at4");
    checkAxisIndex(*this, 2, k, "Tensor::at4");
    checkAxisIndex(*this, 3, l, "Tensor::at4");
    return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}
float Tensor::at4(int i, int j, int k, int l) const {
    checkRank(*this, 4, "Tensor::at4(const)");
    checkAxisIndex(*this, 0, i, "Tensor::at4(const)");
    checkAxisIndex(*this, 1, j, "Tensor::at4(const)");
    checkAxisIndex(*this, 2, k, "Tensor::at4(const)");
    checkAxisIndex(*this, 3, l, "Tensor::at4(const)");
    return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

float* Tensor::rowPtr(int row) {
    checkRank(*this, 2, "Tensor::rowPtr");
    checkAxisIndex(*this, 0, row, "Tensor::rowPtr");
    return data.data() + row * shape[1];
}
const float* Tensor::rowPtr(int row) const {
    checkRank(*this, 2, "Tensor::rowPtr");
    checkAxisIndex(*this, 0, row, "Tensor::rowPtr");
    return data.data() + row * shape[1];
}

bool Tensor::isSameShape(const std::vector<int>& expected,
                         const char* caller) const {
    if (shape != expected) {
        // 错误码：形状不匹配
        BIZLOG(ErrorCode::kTensorShapeMismatch, callerName(caller),
               shapeToString(expected), shapeString());
        return false;
    }
    return true;
}

Tensor Tensor::reshapeChecked(const std::vector<int>& new_shape,
                              const char* caller) const {
    const size_t new_total = checkedNumel(new_shape, callerName(caller));
    if (new_total != data.size()) {
        // 错误码：reshape 前后元素数量不一致
        BIZLOG(ErrorCode::kTensorReshapeSizeMismatch, callerName(caller),
               data.size(), new_total);
        return Tensor();
    }
    Tensor r = *this;
    r.shape = new_shape;
    return r;
}

std::string Tensor::shapeString() const { return shapeToString(shape); }

void Tensor::print(const std::string& name, bool print_data) const {
    // 原先直接写 std::cout，现统一改为经日志输出（info 级，落文件 + 控制台）。
    std::string msg;
    if (!name.empty()) {
        msg += name + " ";
    }
    msg += "shape = " + shapeString() + " size = " + std::to_string(size());
    if (print_data) {
        msg += " data = ";
        for (size_t i = 0; i < size(); ++i) {
            msg += std::to_string(data[i]);
            if (i < size() - 1) {
                msg += " ";
            }
        }
    }
    spdlog::info("{}", msg);
}

Tensor makeTensor1D(int d0, float fill) { return Tensor({d0}, fill); }
Tensor makeTensor2D(int d0, int d1, float fill) {
    return Tensor({d0, d1}, fill);
}
Tensor makeTensor3D(int d0, int d1, int d2, float fill) {
    return Tensor({d0, d1, d2}, fill);
}
Tensor makeTensor4D(int d0, int d1, int d2, int d3, float fill) {
    return Tensor({d0, d1, d2, d3}, fill);
}

}  // namespace mini_llama
