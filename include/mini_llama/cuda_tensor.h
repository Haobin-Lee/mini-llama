// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef MINI_LLAMA_CUDA_TENSOR_H_
#define MINI_LLAMA_CUDA_TENSOR_H_

#include <cstddef>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

class CudaTensor {
public:
    CudaTensor() = default;
    explicit CudaTensor(const std::vector<int>& shape, int device_id = 0);

    CudaTensor(const CudaTensor&) = delete;
    CudaTensor& operator=(const CudaTensor&) = delete;

    CudaTensor(CudaTensor&&) noexcept = default;
    CudaTensor& operator=(CudaTensor&&) noexcept = default;

    void reset();
    void reset(const std::vector<int>& shape, int device_id = 0);

    void uploadFrom(const Tensor& src);
    Tensor download() const;
    void downloadTo(Tensor& dst) const;

    void* data();
    const void* data() const;

    const std::vector<int>& shape() const { return shape_; }
    int numDims() const { return static_cast<int>(shape_.size()); }
    size_t size() const { return numel_; }
    size_t bytes() const { return buffer_.bytes(); }
    int deviceId() const { return device_id_; }
    bool empty() const { return buffer_.empty(); }

    std::string shapeString() const;

private:
    std::vector<int> shape_;
    size_t numel_ = 0;
    int device_id_ = 0;
    CudaDeviceBuffer buffer_;
};

CudaTensor cudaTensorFromHost(const Tensor& src, int device_id = 0);

}  // namespace mini_llama

#endif  // MINI_LLAMA_CUDA_TENSOR_H_