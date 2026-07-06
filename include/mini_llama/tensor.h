// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_TENSOR_H_
#define INCLUDE_MINI_LLAMA_TENSOR_H_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {

// Minimal dense tensor for F32 data. Row-major storage.
struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() = default;
    explicit Tensor(const std::vector<int>& input_shape, float fill = 0.0f);

    size_t size() const { return data.size(); }
    size_t numElements() const { return data.size(); }
    int numDims() const { return static_cast<int>(shape.size()); }

    // Compute flat index from multi-dimensional indices.
    size_t flatIndex(const std::vector<int>& indices) const;

    // Element access by multi-dimensional indices.
    float& at(const std::vector<int>& indices);
    float at(const std::vector<int>& indices) const;

    // Convenience 1D/2D/3D/4D accessors.
    float& at1(int i);
    float at1(int i) const;
    float& at2(int i, int j);
    float at2(int i, int j) const;
    float& at3(int i, int j, int k);
    float at3(int i, int j, int k) const;
    float& at4(int i, int j, int k, int l);
    float at4(int i, int j, int k, int l) const;

    // Pointer to a row of a 2D tensor: rowPtr(r) -> &data[r * cols].
    float* rowPtr(int row);
    const float* rowPtr(int row) const;

    // Throw if shape != expected.
    void assertShape(const std::vector<int>& expected, const char* caller) const;

    // Reshape after verifying element count is unchanged.
    Tensor reshapeChecked(const std::vector<int>& new_shape, const char* caller) const;

    float& operator[](size_t i) { return data[i]; }
    float operator[](size_t i) const { return data[i]; }

    std::string shapeString() const;
    void print(const std::string& name = "", bool print_data = false) const;
};

Tensor makeTensor1D(int d0, float fill = 0.0f);
Tensor makeTensor2D(int d0, int d1, float fill = 0.0f);
Tensor makeTensor3D(int d0, int d1, int d2, float fill = 0.0f);
Tensor makeTensor4D(int d0, int d1, int d2, int d3, float fill = 0.0f);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_TENSOR_H_
