// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/cuda_runtime.h"

#include <sstream>
#include <stdexcept>
#include <string>

#ifdef MINI_LLAMA_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace mini_llama {
namespace {

    std::runtime_error cudaNotBuiltError() {
        return std::runtime_error(
            "CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON "
            "on "
            "a NVIDIA CUDA machine.");
    }

#ifdef MINI_LLAMA_USE_CUDA

    void checkCuda(cudaError_t err, const char* expr) {
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA runtime error in " +
                                     std::string(expr) + ": " +
                                     cudaGetErrorString(err));
        }
    }

    cudaMemcpyKind toCudaMemcpyKind(CudaMemcpyKind kind) {
        switch (kind) {
            case CudaMemcpyKind::kHostToDevice:
                return cudaMemcpyHostToDevice;
            case CudaMemcpyKind::kDeviceToHost:
                return cudaMemcpyDeviceToHost;
            case CudaMemcpyKind::kDeviceToDevice:
                return cudaMemcpyDeviceToDevice;
        }
        throw std::runtime_error("unknown CUDA memcpy kind");
    }

#endif  // MINI_LLAMA_USE_CUDA

}  // namespace

bool cudaRuntimeBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

int cudaDeviceCount() {
#ifdef MINI_LLAMA_USE_CUDA
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
#else
    throw cudaNotBuiltError();
#endif
}

void cudaSetDeviceId(int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
#else
    (void)device_id;
    throw cudaNotBuiltError();
#endif
}

CudaDeviceInfo cudaGetDeviceInfo(int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    int count = cudaDeviceCount();
    if (device_id < 0 || device_id >= count) {
        throw std::runtime_error("CUDA device " + std::to_string(device_id) +
                                 " is not available; detected " +
                                 std::to_string(count) + " device(s)");
    }

    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, device_id),
              "cudaGetDeviceProperties");

    CudaDeviceInfo info;
    info.id = device_id;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.total_memory_bytes = prop.totalGlobalMem;
    checkCuda(cudaDriverGetVersion(&info.driver_version),
              "cudaDriverGetVersion");
    checkCuda(cudaRuntimeGetVersion(&info.runtime_version),
              "cudaRuntimeGetVersion");
    return info;
#else
    (void)device_id;
    throw cudaNotBuiltError();
#endif
}

std::string cudaFormatDeviceInfo(const CudaDeviceInfo& info) {
    double total_gb = static_cast<double>(info.total_memory_bytes) /
                      (1024.0 * 1024.0 * 1024.0);
    std::ostringstream out;
    out << "device " << info.id << ": " << info.name << ", compute capability "
        << info.compute_major << "." << info.compute_minor << ", total memory "
        << total_gb << " GB"
        << ", cuda runtime " << info.runtime_version << ", driver "
        << info.driver_version;
    return out.str();
}

void* cudaMallocBytes(size_t bytes) {
#ifdef MINI_LLAMA_USE_CUDA
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
#else
    (void)bytes;
    throw cudaNotBuiltError();
#endif
}

void cudaFreeBytes(void* ptr) {
#ifdef MINI_LLAMA_USE_CUDA
    if (ptr == nullptr) {
        return;
    }
    checkCuda(cudaFree(ptr), "cudaFree");
#else
    (void)ptr;
    throw cudaNotBuiltError();
#endif
}

void cudaMemcpyBytes(void* dst, const void* src, size_t bytes,
                     CudaMemcpyKind kind) {
#ifdef MINI_LLAMA_USE_CUDA
    if (bytes == 0) {
        return;
    }
    if (dst == nullptr || src == nullptr) {
        throw std::runtime_error(
            "cudaMemcpy requires non-null src and dst for non-empty copies");
    }
    checkCuda(cudaMemcpy(dst, src, bytes, toCudaMemcpyKind(kind)),
              "cudaMemcpy");
#else
    (void)dst;
    (void)src;
    (void)bytes;
    (void)kind;
    throw cudaNotBuiltError();
#endif
}

CudaDeviceBuffer::CudaDeviceBuffer(size_t bytes, int device_id) {
    reset(bytes, device_id);
}

CudaDeviceBuffer::~CudaDeviceBuffer() { releaseNoexcept(); }

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_), device_id_(other.device_id_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
    other.device_id_ = 0;
}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(
    CudaDeviceBuffer&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    releaseNoexcept();
    data_ = other.data_;
    bytes_ = other.bytes_;
    device_id_ = other.device_id_;
    other.data_ = nullptr;
    other.bytes_ = 0;
    other.device_id_ = 0;
    return *this;
}

void CudaDeviceBuffer::reset() {
    if (data_ == nullptr) {
        return;
    }
    cudaSetDeviceId(device_id_);
    cudaFreeBytes(data_);
    data_ = nullptr;
    bytes_ = 0;
    device_id_ = 0;
}

void CudaDeviceBuffer::reset(size_t bytes, int device_id) {
    reset();
    if (bytes == 0) {
        device_id_ = device_id;
        return;
    }
    cudaSetDeviceId(device_id);
    data_ = cudaMallocBytes(bytes);
    bytes_ = bytes;
    device_id_ = device_id;
}

void CudaDeviceBuffer::upload(const void* src, size_t bytes) {
    if (bytes > bytes_) {
        throw std::runtime_error("CUDA upload exceeds device buffer size");
    }
    cudaSetDeviceId(device_id_);
    cudaMemcpyBytes(data_, src, bytes, CudaMemcpyKind::kHostToDevice);
}

void CudaDeviceBuffer::download(void* dst, size_t bytes) const {
    if (bytes > bytes_) {
        throw std::runtime_error("CUDA download exceeds device buffer size");
    }
    cudaSetDeviceId(device_id_);
    cudaMemcpyBytes(dst, data_, bytes, CudaMemcpyKind::kDeviceToHost);
}

void CudaDeviceBuffer::releaseNoexcept() noexcept {
    try {
        reset();
    } catch (...) {
    }
}

}  // namespace mini_llama