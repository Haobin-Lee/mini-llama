// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/cuda_matmul.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"

#ifdef MINI_LLAMA_USE_CUDA
#include <cublas_v2.h>
#endif

namespace mini_llama {
namespace {

    std::runtime_error cudaMatmulNotBuiltError() {
        return std::runtime_error(
            "CUDA Matmul was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON "
            "on a "
            "NVIDIA CUDA machine.");
    }

#ifdef MINI_LLAMA_USE_CUDA

    // ---------------------------------------------------------------------------
    // cuBLAS error handling
    // ---------------------------------------------------------------------------

    const char* cublasStatusName(cublasStatus_t status) {
        switch (status) {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
        }
        return "CUBLAS_STATUS_UNKNOWN";
    }

    void checkCublas(cublasStatus_t status, const char* expr) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS error in " + std::string(expr) +
                                     ": " + cublasStatusName(status));
        }
    }

    // RAII wrapper for cublasHandle_t. Creates handle on construction,
    // destroys on destruction. Not copyable.
    class CublasHandle {
    public:
        explicit CublasHandle(int device_id) {
            cudaSetDeviceId(device_id);
            checkCublas(cublasCreate(&handle_), "cublasCreate");
        }

        ~CublasHandle() {
            if (handle_ != nullptr) {
                cublasDestroy(handle_);
            }
        }

        CublasHandle(const CublasHandle&) = delete;
        CublasHandle& operator=(const CublasHandle&) = delete;

        cublasHandle_t get() const { return handle_; }

    private:
        cublasHandle_t handle_ = nullptr;
    };

    // ---------------------------------------------------------------------------
    // Shape validation
    // ---------------------------------------------------------------------------

    void validateMatmulShapes(const Tensor& a, const Tensor& b) {
        if (a.numDims() != 2 || b.numDims() != 2) {
            throw std::runtime_error(
                "cudaMatmul: expected a and b to be 2D tensors");
        }
        if (a.shape[1] != b.shape[0]) {
            throw std::runtime_error("cudaMatmul: dimension mismatch " +
                                     a.shapeString() + " vs " +
                                     b.shapeString());
        }
    }

    // Extracts in_features from x shape: [in_features] or [batch, in_features].
    int linearInFeatures(const std::vector<int>& x_shape,
                         const std::string& x_shape_str, const char* caller) {
        if (x_shape.size() == 1) {
            return x_shape[0];
        }
        if (x_shape.size() == 2) {
            return x_shape[1];
        }
        throw std::runtime_error(
            std::string(caller) +
            ": expected x shape [in_features] or [batch, in_features], got " +
            x_shape_str);
    }

    void validateLinearShape(const std::vector<int>& x_shape,
                             const std::string& x_shape_str,
                             const std::vector<int>& w_shape,
                             const char* caller) {
        if (w_shape.size() != 2) {
            throw std::runtime_error(
                std::string(caller) +
                ": expected weight shape [out_features, in_features]");
        }
        int in_features = linearInFeatures(x_shape, x_shape_str, caller);
        if (w_shape[1] != in_features) {
            throw std::runtime_error(std::string(caller) +
                                     ": dimension mismatch x=" + x_shape_str +
                                     " weight=[" + std::to_string(w_shape[0]) +
                                     ", " + std::to_string(w_shape[1]) + "]");
        }
    }

    void validateLinearInputs(const Tensor& x,
                              const std::vector<int>& w_shape) {
        validateLinearShape(x.shape, x.shapeString(), w_shape, "cudaLinear");
    }

    // CPU bias addition: y[row, col] += bias[col].
    void addBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols) {
        if (bias.numDims() != 1 || bias.shape[0] != cols) {
            throw std::runtime_error(
                "cudaLinear: expected bias shape [out_features]");
        }
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                y.data[row * cols + col] += bias.data[col];
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Internal implementation: Linear with device weight pointer
    // ---------------------------------------------------------------------------

    // Host input, host output. Uploads x, runs Sgemm, downloads y.
    Tensor cudaLinearWithDevicePointer(const Tensor& x, const void* w_device,
                                       const std::vector<int>& w_shape,
                                       const Tensor* bias, int device_id) {
        if (w_device == nullptr) {
            throw std::runtime_error(
                "cudaLinear: device weight pointer is null");
        }
        validateLinearInputs(x, w_shape);

        bool input_is_1d = x.numDims() == 1;
        int rows = input_is_1d ? 1 : x.shape[0];
        int in_features = input_is_1d ? x.shape[0] : x.shape[1];
        int out_features = w_shape[0];

        Tensor y(input_is_1d ? std::vector<int>{out_features}
                             : std::vector<int>{rows, out_features},
                 0.0f);

        CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
        CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
        x_dev.upload(x.data.data(), x.size() * sizeof(float));

        CublasHandle handle(device_id);
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Row-major y = x[rows,k] @ W[out,k]^T
        // In column-major (cuBLAS): y^T[out,rows] = W[out,k] * x^T[k,rows]
        checkCublas(
            cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, out_features,
                        rows, in_features, &alpha,
                        static_cast<const float*>(w_device), in_features,
                        static_cast<const float*>(x_dev.data()), in_features,
                        &beta, static_cast<float*>(y_dev.data()), out_features),
            "cublasSgemm(cudaLinear)");

        y_dev.download(y.data.data(), y.size() * sizeof(float));

        if (bias != nullptr) {
            addBiasInPlace(y, *bias, rows, out_features);
        }
        return y;
    }

    // Device input, device output. No upload/download.
    CudaTensor cudaLinearDeviceInputImpl(const CudaTensor& x,
                                         const void* w_device,
                                         const std::vector<int>& w_shape,
                                         int device_id) {
        if (w_device == nullptr) {
            throw std::runtime_error(
                "cudaLinearDeviceInput: device weight pointer is null");
        }
        if (x.deviceId() != device_id) {
            throw std::runtime_error(
                "cudaLinearDeviceInput: input tensor is on a different CUDA "
                "device");
        }
        validateLinearShape(x.shape(), x.shapeString(), w_shape,
                            "cudaLinearDeviceInput");

        bool input_is_1d = x.numDims() == 1;
        int rows = input_is_1d ? 1 : x.shape()[0];
        int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
        int out_features = w_shape[0];

        CudaTensor y(input_is_1d ? std::vector<int>{out_features}
                                 : std::vector<int>{rows, out_features},
                     device_id);

        CublasHandle handle(device_id);
        const float alpha = 1.0f;
        const float beta = 0.0f;

        checkCublas(
            cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, out_features,
                        rows, in_features, &alpha,
                        static_cast<const float*>(w_device), in_features,
                        static_cast<const float*>(x.data()), in_features, &beta,
                        static_cast<float*>(y.data()), out_features),
            "cublasSgemm(cudaLinearDeviceInput)");

        return y;
    }

#endif  // MINI_LLAMA_USE_CUDA

}  // namespace

// ===========================================================================
// Public API
// ===========================================================================

bool cudaMatmulBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
    return true;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// cudaMatmul: c = a @ b
// a: [m, k], b: [k, n] -> c: [m, n]
// ---------------------------------------------------------------------------
Tensor cudaMatmul(const Tensor& a, const Tensor& b, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    validateMatmulShapes(a, b);

    int m = a.shape[0];
    int k = a.shape[1];
    int n = b.shape[1];
    Tensor c({m, n}, 0.0f);

    CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
    CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
    CudaDeviceBuffer c_dev(c.size() * sizeof(float), device_id);
    a_dev.upload(a.data.data(), a.size() * sizeof(float));
    b_dev.upload(b.data.data(), b.size() * sizeof(float));

    CublasHandle handle(device_id);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major c[m,n] = a[m,k] @ b[k,n]
    // Column-major equivalent: c^T[n,m] = b^T[n,k] @ a^T[k,m]
    // The row-major buffer for c has the same byte layout as column-major c^T.
    checkCublas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                            &alpha, static_cast<const float*>(b_dev.data()), n,
                            static_cast<const float*>(a_dev.data()), k, &beta,
                            static_cast<float*>(c_dev.data()), n),
                "cublasSgemm(cudaMatmul)");

    c_dev.download(c.data.data(), c.size() * sizeof(float));
    return c;
#else
    (void)a;
    (void)b;
    (void)device_id;
    throw cudaMatmulNotBuiltError();
#endif
}

// ---------------------------------------------------------------------------
// cudaLinear: y = x @ W^T + bias
// ---------------------------------------------------------------------------
Tensor cudaLinear(const Tensor& x, const Tensor& weight, const Tensor* bias,
                  int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    if (weight.numDims() != 2) {
        throw std::runtime_error(
            "cudaLinear: expected weight shape [out_features, in_features]");
    }
    CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
    w_dev.upload(weight.data.data(), weight.size() * sizeof(float));
    return cudaLinearWithDevicePointer(x, w_dev.data(), weight.shape, bias,
                                       device_id);
#else
    (void)x;
    (void)weight;
    (void)bias;
    (void)device_id;
    throw cudaMatmulNotBuiltError();
#endif
}

// ---------------------------------------------------------------------------
// cudaLinearDeviceWeight: weight already on GPU
// ---------------------------------------------------------------------------
Tensor cudaLinearDeviceWeight(const Tensor& x, const void* w_device,
                              const std::vector<int>& w_shape,
                              const Tensor* bias, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    return cudaLinearWithDevicePointer(x, w_device, w_shape, bias, device_id);
#else
    (void)x;
    (void)w_device;
    (void)w_shape;
    (void)bias;
    (void)device_id;
    throw cudaMatmulNotBuiltError();
#endif
}

// ---------------------------------------------------------------------------
// cudaLinearDeviceInput: both input and output on GPU
// ---------------------------------------------------------------------------
CudaTensor cudaLinearDeviceInput(const CudaTensor& x, const void* w_device,
                                 const std::vector<int>& w_shape,
                                 int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
    return cudaLinearDeviceInputImpl(x, w_device, w_shape, device_id);
#else
    (void)x;
    (void)w_device;
    (void)w_shape;
    (void)device_id;
    throw cudaMatmulNotBuiltError();
#endif
}

}  // namespace mini_llama