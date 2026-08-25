// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef MINI_LLAMA_CUDA_RUNTIME_H_
#define MINI_LLAMA_CUDA_RUNTIME_H_

#include <cstddef>
#include <string>

namespace mini_llama {

struct CudaDeviceInfo {
    int id = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    size_t total_memory_bytes = 0;
    int driver_version = 0;
    int runtime_version = 0;
};

enum class CudaMemcpyKind {
    kHostToDevice,
    kDeviceToHost,
    kDeviceToDevice,
};

bool cudaRuntimeBuilt();
int cudaDeviceCount();
void cudaSetDeviceId(int device_id);
CudaDeviceInfo cudaGetDeviceInfo(int device_id);
std::string cudaFormatDeviceInfo(const CudaDeviceInfo& info);

void* cudaMallocBytes(size_t bytes);
void cudaFreeBytes(void* ptr);
void cudaMemcpyBytes(void* dst, const void* src, size_t bytes,
                     CudaMemcpyKind kind);

class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() = default;
    explicit CudaDeviceBuffer(size_t bytes, int device_id = 0);
    ~CudaDeviceBuffer();

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

    void reset();
    void reset(size_t bytes, int device_id = 0);

    void upload(const void* src, size_t bytes);
    void download(void* dst, size_t bytes) const;

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t bytes() const { return bytes_; }
    int device_id() const { return device_id_; }
    bool empty() const { return data_ == nullptr; }

private:
    void releaseNoexcept() noexcept;

    void* data_ = nullptr;
    size_t bytes_ = 0;
    int device_id_ = 0;
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_CUDA_RUNTIME_H_