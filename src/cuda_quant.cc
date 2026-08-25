#include "mini_llama/cuda_quant.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace mini_llama {
namespace {

    std::runtime_error cudaQuantNotBuiltError() {
        return std::runtime_error(
            "CUDA quant was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON "
            "on a "
            "NVIDIA CUDA machine.");
    }

}  // namespace

bool cudaQuantBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA

Tensor cudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
    (void)x;
    (void)weight;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cudaQuantNotBuiltError();
}

Tensor cudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device,
                                 size_t q8_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
    (void)x;
    (void)q8_0_device;
    (void)q8_0_block_count;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cudaQuantNotBuiltError();
}

CudaTensor cudaQ80LinearDeviceInput(const CudaTensor& x,
                                    const void* q8_0_device,
                                    size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
    (void)x;
    (void)q8_0_device;
    (void)q8_0_block_count;
    (void)weight_shape;
    (void)device_id;
    throw cudaQuantNotBuiltError();
}

Tensor cudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
    (void)x;
    (void)weight;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cudaQuantNotBuiltError();
}

Tensor cudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device,
                                 size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
    (void)x;
    (void)q4_0_device;
    (void)q4_0_block_count;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cudaQuantNotBuiltError();
}

CudaTensor cudaQ40LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_0_device,
                                    size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
    (void)x;
    (void)q4_0_device;
    (void)q4_0_block_count;
    (void)weight_shape;
    (void)device_id;
    throw cudaQuantNotBuiltError();
}

Tensor cudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
    (void)x;
    (void)weight;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cudaQuantNotBuiltError();
}

Tensor cudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device,
                                 size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
    (void)x;
    (void)q4_1_device;
    (void)q4_1_block_count;
    (void)weight_shape;
    (void)bias;
    (void)device_id;
    throw cudaQuantNotBuiltError();
}

CudaTensor cudaQ41LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_1_device,
                                    size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
    (void)x;
    (void)q4_1_device;
    (void)q4_1_block_count;
    (void)weight_shape;
    (void)device_id;
    throw cudaQuantNotBuiltError();
}

#endif  // MINI_LLAMA_USE_CUDA

}  // namespace mini_llama