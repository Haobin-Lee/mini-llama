// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// CUDA 后端测试：runtime/buffer/tensor、ops、matmul、quant、kv cache、
// attention（对照组实现）以及 CPU/CUDA 端到端一致性。
//
// 可移植性约定：
//   - 纯 CPU 构建或无 GPU 环境下所有 CUDA 用例自动 skip（返回 true）。
//   - 数值对照统一使用最大绝对误差断言，错误时会打印误差与首个超标位置。

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/cuda_attention.h"
#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/cuda_matmul.h"
#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_quant.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/ops.h"
#include "mini_llama/quant.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"
#include "tests/test_main.h"

using namespace mini_llama;

namespace {

// ---------------------------------------------------------------------------
// 公共辅助
// ---------------------------------------------------------------------------

bool cudaAvailable() { return cudaRuntimeBuilt() && cudaDeviceCount() > 0; }

// 返回 true 表示当前环境没有可用 CUDA，测试应跳过
bool skipWithoutCuda(const char* test_name) {
    if (cudaAvailable()) {
        return false;
    }
    std::cout << "  (skip " << test_name << ": CUDA not built or no device)\n";
    return true;
}

template <typename Ex>
bool throwsAs(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const Ex&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// 用确定性线性序列填充张量
void fillLinear(Tensor& t, float scale, float offset) {
    for (size_t i = 0; i < t.size(); ++i) {
        t.data[i] = offset + scale * static_cast<float>(i);
    }
}

Tensor makeLinear(int n, float scale, float offset) {
    Tensor t({n}, 0.0f);
    fillLinear(t, scale, offset);
    return t;
}

// 断言两个张量形状一致且最大绝对误差小于 tol
bool expectNear(const Tensor& actual, const Tensor& expected, float tol,
                const char* what) {
    if (actual.shape != expected.shape) {
        std::cerr << "  shape mismatch for " << what
                  << ": actual=" << actual.shapeString()
                  << " expected=" << expected.shapeString() << std::endl;
        return false;
    }
    float max_diff = 0.0f;
    size_t arg = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::fabs(actual.data[i] - expected.data[i]);
        if (diff > max_diff) {
            max_diff = diff;
            arg = i;
        }
    }
    if (max_diff > tol) {
        std::cerr << "  " << what << ": max diff " << max_diff << " at index "
                  << arg << " (actual=" << actual.data[arg]
                  << ", expected=" << expected.data[arg] << ", tol=" << tol
                  << ")" << std::endl;
        return false;
    }
    return true;
}

float maxAbsDiff(const Tensor& a, const Tensor& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a.data[i] - b.data[i]));
    }
    return max_diff;
}

// Q4_1 没有 CPU 量化入口，手工构造确定的 block（d=0.5 / m=0.0）
std::vector<BlockQ41> makeQ41Blocks(int out_features, int in_features) {
    const int blocks_per_row =
        (in_features + kQ41BlockSize - 1) / kQ41BlockSize;
    std::vector<BlockQ41> blocks(static_cast<size_t>(out_features) *
                                 static_cast<size_t>(blocks_per_row));
    for (auto& block : blocks) {
        block.d = 0x3800;  // fp16(0.5)
        block.m = 0x0000;  // fp16(0.0)
        for (int i = 0; i < 16; ++i) {
            int lo = (i * 3 + 1) & 0x0F;
            int hi = (i * 5 + 2) & 0x0F;
            block.qs[i] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }
    return blocks;
}

// attention 的 CPU 参考实现：softmax(q·k^T / sqrt(d)) · v
// K/V 直接从 CudaKvCache 读回，GQA 映射与内核一致：kv_head = head / group
Tensor attentionReference(const Tensor& q, const CudaKvCache& cache, int layer,
                          int pos, int n_heads, int n_kv_heads, int head_dim) {
    Tensor out({n_heads, head_dim}, 0.0f);
    const int group = n_heads / n_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int h = 0; h < n_heads; ++h) {
        const int kv_head = h / group;
        std::vector<float> scores(static_cast<size_t>(pos) + 1, 0.0f);
        float max_score = -std::numeric_limits<float>::infinity();

        for (int t = 0; t <= pos; ++t) {
            Tensor key = cache.readKeyHead(layer, t, kv_head);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot +=
                    q.data[static_cast<size_t>(h) * head_dim + d] * key.data[d];
            }
            scores[static_cast<size_t>(t)] = dot * scale;
            max_score = std::max(max_score, scores[static_cast<size_t>(t)]);
        }

        float denom = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            scores[static_cast<size_t>(t)] =
                std::exp(scores[static_cast<size_t>(t)] - max_score);
            denom += scores[static_cast<size_t>(t)];
        }

        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                Tensor value = cache.readValueHead(layer, t, kv_head);
                acc += scores[static_cast<size_t>(t)] / denom * value.data[d];
            }
            out.at2(h, d) = acc;
        }
    }
    return out;
}

}  // namespace

// ===========================================================================
// runtime / device buffer / CudaTensor
// ===========================================================================

static bool testCudaRuntimeBuiltFlag() {
    const bool built = cudaRuntimeBuilt();
    if (!built) {
        // CPU-only 构建：任何 CUDA 入口都应抛 not-built 错误
        MINI_LLAMA_ASSERT_TRUE(
            throwsAs<std::runtime_error>([] { (void)cudaDeviceCount(); }));
        return true;
    }
    MINI_LLAMA_ASSERT_TRUE(cudaDeviceCount() >= 0);
    return true;
}

static bool testCudaDeviceInfo() {
    if (skipWithoutCuda("cuda_device_info")) {
        return true;
    }
    const int count = cudaDeviceCount();
    MINI_LLAMA_ASSERT_TRUE(count >= 1);

    CudaDeviceInfo info = cudaGetDeviceInfo(0);
    MINI_LLAMA_ASSERT_EQ(info.id, 0);
    MINI_LLAMA_ASSERT_TRUE(!info.name.empty());
    MINI_LLAMA_ASSERT_TRUE(info.compute_major > 0);
    MINI_LLAMA_ASSERT_TRUE(info.total_memory_bytes > 0);
    MINI_LLAMA_ASSERT_TRUE(info.runtime_version > 0);

    const std::string text = cudaFormatDeviceInfo(info);
    MINI_LLAMA_ASSERT_TRUE(text.find("device 0") != std::string::npos);

    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([] { (void)cudaGetDeviceInfo(-1); }));
    cudaSetDeviceId(0);
    return true;
}

static bool testCudaDeviceBufferRoundtrip() {
    if (skipWithoutCuda("cuda_device_buffer_roundtrip")) {
        return true;
    }

    const size_t bytes = 4 * sizeof(float);
    CudaDeviceBuffer buffer(bytes, 0);
    MINI_LLAMA_ASSERT_TRUE(!buffer.empty());
    MINI_LLAMA_ASSERT_EQ(buffer.bytes(), bytes);
    MINI_LLAMA_ASSERT_EQ(buffer.device_id(), 0);

    const std::vector<float> src = {1.0f, -2.0f, 3.5f, 4.25f};
    buffer.upload(src.data(), bytes);

    std::vector<float> dst(4, 0.0f);
    buffer.download(dst.data(), bytes);
    for (size_t i = 0; i < src.size(); ++i) {
        MINI_LLAMA_ASSERT_NEAR(dst[i], src[i], 1e-6f);
    }

    // 超出缓冲区大小的拷贝必须被拒绝
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { buffer.upload(src.data(), bytes + sizeof(float)); }));
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { buffer.download(dst.data(), bytes + sizeof(float)); }));

    // 移动语义：源 buffer 置空
    CudaDeviceBuffer moved(std::move(buffer));
    MINI_LLAMA_ASSERT_TRUE(buffer.empty());
    MINI_LLAMA_ASSERT_EQ(buffer.bytes(), static_cast<size_t>(0));
    MINI_LLAMA_ASSERT_EQ(moved.bytes(), bytes);
    moved.reset();
    MINI_LLAMA_ASSERT_TRUE(moved.empty());

    CudaDeviceBuffer zero(0, 0);
    MINI_LLAMA_ASSERT_TRUE(zero.empty());
    zero.reset(2 * sizeof(float), 0);
    MINI_LLAMA_ASSERT_TRUE(!zero.empty());
    return true;
}

static bool testCudaTensorRoundtrip() {
    if (skipWithoutCuda("cuda_tensor_roundtrip")) {
        return true;
    }

    CudaTensor device_tensor({2, 3}, 0);
    MINI_LLAMA_ASSERT_EQ(device_tensor.numDims(), 2);
    MINI_LLAMA_ASSERT_EQ(device_tensor.size(), static_cast<size_t>(6));
    MINI_LLAMA_ASSERT_EQ(device_tensor.bytes(), 6 * sizeof(float));
    MINI_LLAMA_ASSERT_TRUE(device_tensor.shape() == std::vector<int>({2, 3}));
    MINI_LLAMA_ASSERT_TRUE(!device_tensor.empty());

    Tensor host({2, 3}, 0.0f);
    fillLinear(host, 0.5f, -1.0f);
    device_tensor.uploadFrom(host);

    Tensor back = device_tensor.download();
    MINI_LLAMA_ASSERT_TRUE(expectNear(back, host, 1e-6f, "CudaTensor 回读"));

    Tensor dst({2, 3}, 0.0f);
    device_tensor.downloadTo(dst);
    MINI_LLAMA_ASSERT_TRUE(expectNear(dst, host, 1e-6f, "downloadTo"));

    // 形状不一致必须报错
    Tensor wrong({3, 2}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([&] { device_tensor.uploadFrom(wrong); }));
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([&] { device_tensor.downloadTo(wrong); }));

    CudaTensor from_host = cudaTensorFromHost(host, 0);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(from_host.download(), host, 1e-6f, "cudaTensorFromHost"));

    device_tensor.reset();
    MINI_LLAMA_ASSERT_TRUE(device_tensor.empty());
    return true;
}

// ===========================================================================
// ops：RMSNorm / SiLU / Elementwise / Softmax / Embedding / RoPE
// ===========================================================================

static bool testCudaRmsNormMatchesCpu() {
    if (skipWithoutCuda("cuda_rms_norm_matches_cpu")) {
        return true;
    }
    // 8/33/64 覆盖：整 block、非 32 倍数、多 block
    for (int n : {8, 33, 64}) {
        Tensor x = makeLinear(n, 0.1f, -1.0f);
        Tensor w = makeLinear(n, 0.05f, 0.5f);

        Tensor cpu = rmsNorm(x, w, 1e-5f);
        Tensor gpu = cudaRmsNorm(x, w, 1e-5f);
        MINI_LLAMA_ASSERT_TRUE(expectNear(gpu, cpu, 2e-4f, "cudaRmsNorm"));
    }

    Tensor x({4}, 1.0f);
    Tensor w({4}, 1.0f);
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([&] { (void)cudaRmsNorm(x, w, 0.0f); }));
    Tensor w_bad({3}, 1.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { (void)cudaRmsNorm(x, w_bad, 1e-5f); }));
    Tensor x_2d({1, 4}, 1.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { (void)cudaRmsNorm(x_2d, w, 1e-5f); }));
    return true;
}

static bool testCudaSiluMatchesCpu() {
    if (skipWithoutCuda("cuda_silu_matches_cpu")) {
        return true;
    }
    Tensor x = makeLinear(17, 0.3f, -2.0f);
    Tensor cpu = silu(x);
    Tensor gpu = cudaSilu(x);
    MINI_LLAMA_ASSERT_TRUE(expectNear(gpu, cpu, 1e-5f, "cudaSilu"));
    return true;
}

static bool testCudaElementwiseMatchesCpu() {
    if (skipWithoutCuda("cuda_elementwise_matches_cpu")) {
        return true;
    }
    Tensor a = makeLinear(9, 0.2f, -1.0f);
    Tensor b = makeLinear(9, -0.1f, 0.5f);

    Tensor cpu_mul = elementwiseMul(a, b);
    Tensor gpu_mul = cudaElementwiseMul(a, b);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(gpu_mul, cpu_mul, 1e-6f, "cudaElementwiseMul"));

    Tensor expected_add({9}, 0.0f);
    for (int i = 0; i < 9; ++i) {
        expected_add[i] = a[i] + b[i];
    }
    Tensor gpu_add = cudaElementwiseAdd(a, b);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(gpu_add, expected_add, 1e-6f, "cudaElementwiseAdd"));

    Tensor b_wrong({4}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { (void)cudaElementwiseMul(a, b_wrong); }));
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { (void)cudaElementwiseAdd(a, b_wrong); }));
    return true;
}

static bool testCudaSoftmaxMatchesCpu() {
    if (skipWithoutCuda("cuda_softmax_matches_cpu")) {
        return true;
    }
    // 5（小于 block）与 100（跨线程 stride）两种规模
    for (int n : {5, 100}) {
        Tensor x = makeLinear(n, 0.2f, -1.5f);
        Tensor cpu = softmax(x);
        Tensor gpu = cudaSoftmax(x);

        MINI_LLAMA_ASSERT_TRUE(expectNear(gpu, cpu, 1e-5f, "cudaSoftmax"));
        float sum = 0.0f;
        for (size_t i = 0; i < gpu.size(); ++i) {
            sum += gpu.data[i];
        }
        MINI_LLAMA_ASSERT_NEAR(sum, 1.0f, 1e-4f);
    }

    Tensor x_2d({2, 3}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([&] { (void)cudaSoftmax(x_2d); }));
    return true;
}

static bool testCudaOpsDeviceVariantsMatchHost() {
    if (skipWithoutCuda("cuda_ops_device_variants_match_host")) {
        return true;
    }

    Tensor x = makeLinear(12, 0.15f, -0.8f);
    Tensor w = makeLinear(12, 0.07f, 0.3f);
    Tensor y = makeLinear(12, -0.05f, 0.2f);

    Tensor host_rms = cudaRmsNorm(x, w, 1e-5f);
    Tensor host_silu = cudaSilu(x);

    // 权重常驻显存
    CudaDeviceBuffer w_dev(w.size() * sizeof(float), 0);
    w_dev.upload(w.data.data(), w.size() * sizeof(float));
    CudaDeviceBuffer y_dev(y.size() * sizeof(float), 0);
    y_dev.upload(y.data.data(), y.size() * sizeof(float));

    CudaTensor x_dev = cudaTensorFromHost(x, 0);
    CudaTensor w_dev_tensor = cudaTensorFromHost(w, 0);

    CudaTensor rms_weight =
        cudaRmsNormDeviceWeight(x_dev, w_dev.data(), w.shape, 1e-5f, 0);
    MINI_LLAMA_ASSERT_TRUE(expectNear(rms_weight.download(), host_rms, 2e-4f,
                                      "rmsNormDeviceWeight"));

    CudaTensor rms_input = cudaRmsNormDeviceInput(x_dev, w, 1e-5f, 0);
    MINI_LLAMA_ASSERT_TRUE(expectNear(rms_input.download(), host_rms, 2e-4f,
                                      "rmsNormDeviceInput"));

    CudaTensor silu_dev = cudaSiluDeviceInput(x_dev, 0);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(silu_dev.download(), host_silu, 1e-5f, "siluDeviceInput"));

    Tensor host_mul = cudaElementwiseMul(x, w);
    CudaTensor mul_dev = cudaElementwiseMulDeviceInput(x_dev, w_dev_tensor, 0);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(mul_dev.download(), host_mul, 1e-6f, "ewMulDeviceInput"));

    Tensor host_add = cudaElementwiseAdd(x, y);
    CudaTensor add_dev =
        cudaElementwiseAddDeviceInput(x_dev, cudaTensorFromHost(y, 0), 0);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(add_dev.download(), host_add, 1e-6f, "ewAddDeviceInput"));

    CudaTensor add_weight =
        cudaElementwiseAddDeviceWeight(x_dev, y_dev.data(), y.shape, 0);
    MINI_LLAMA_ASSERT_TRUE(expectNear(add_weight.download(), host_add, 1e-6f,
                                      "ewAddDeviceWeight"));

    // 设备指针为空 / 形状不匹配必须报错
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>([&] {
        (void)cudaRmsNormDeviceWeight(x_dev, nullptr, w.shape, 1e-5f, 0);
    }));
    Tensor w_bad_shape({5}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>([&] {
        (void)cudaElementwiseAddDeviceWeight(x_dev, y_dev.data(),
                                             w_bad_shape.shape, 0);
    }));
    return true;
}

static bool testCudaEmbeddingLookup() {
    if (skipWithoutCuda("cuda_embedding_lookup")) {
        return true;
    }

    Tensor embedding({4, 3}, 0.0f);
    fillLinear(embedding, 0.25f, -1.0f);

    CudaDeviceBuffer emb_dev(embedding.size() * sizeof(float), 0);
    emb_dev.upload(embedding.data.data(), embedding.size() * sizeof(float));

    CudaTensor row =
        cudaEmbeddingLookupDeviceWeight(emb_dev.data(), embedding.shape, 2, 0);
    Tensor row_host = row.download();
    MINI_LLAMA_ASSERT_TRUE(row_host.shape == std::vector<int>({3}));
    for (int d = 0; d < 3; ++d) {
        MINI_LLAMA_ASSERT_NEAR(row_host[d], embedding.at2(2, d), 1e-6f);
    }

    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::out_of_range>([&] {
        (void)cudaEmbeddingLookupDeviceWeight(emb_dev.data(), embedding.shape,
                                              4, 0);
    }));
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>([&] {
        (void)cudaEmbeddingLookupDeviceWeight(nullptr, embedding.shape, 0, 0);
    }));
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>([&] {
        (void)cudaEmbeddingLookupDeviceWeight(emb_dev.data(), {4}, 0, 0);
    }));
    return true;
}

static bool testCudaRopeMatchesCpu() {
    if (skipWithoutCuda("cuda_rope_matches_cpu")) {
        return true;
    }

    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const int pos = 3;
    const float theta = 10000.0f;

    for (RopeType type : {RopeType::kNormal, RopeType::kNeoX}) {
        Tensor q({n_heads, head_dim}, 0.0f);
        Tensor k({n_kv_heads, head_dim}, 0.0f);
        fillLinear(q, 0.11f, -0.7f);
        fillLinear(k, -0.13f, 0.4f);

        Tensor cpu_q = q;
        Tensor cpu_k = k;
        MINI_LLAMA_ASSERT_TRUE(rope(cpu_q, cpu_k, pos, theta, type));

        Tensor gpu_q = q;
        Tensor gpu_k = k;
        cudaRope(gpu_q, gpu_k, pos, theta, type, 0);

        MINI_LLAMA_ASSERT_TRUE(expectNear(gpu_q, cpu_q, 1e-5f, "cudaRope(q)"));
        MINI_LLAMA_ASSERT_TRUE(expectNear(gpu_k, cpu_k, 1e-5f, "cudaRope(k)"));
    }

    Tensor q({2, 4}, 0.0f);
    Tensor k({2, 4}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::out_of_range>(
        [&] { cudaRope(q, k, -1, theta, RopeType::kNormal, 0); }));
    Tensor q_1d({4}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { cudaRope(q_1d, k, 0, theta, RopeType::kNormal, 0); }));
    Tensor q_odd({2, 3}, 0.0f);
    Tensor k_odd({2, 3}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { cudaRope(q_odd, k_odd, 0, theta, RopeType::kNormal, 0); }));
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { cudaRope(q, k, 0, 0.0f, RopeType::kNormal, 0); }));
    return true;
}

// ===========================================================================
// matmul / linear
// ===========================================================================

static bool testCudaMatmulMatchesCpu() {
    if (skipWithoutCuda("cuda_matmul_matches_cpu")) {
        return true;
    }

    // 2x2
    Tensor a({2, 2}, 0.0f);
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    a[3] = 4.0f;
    Tensor b({2, 2}, 0.0f);
    b[0] = 5.0f;
    b[1] = 6.0f;
    b[2] = 7.0f;
    b[3] = 8.0f;
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(cudaMatmul(a, b), matmul(a, b), 1e-5f, "cudaMatmul 2x2"));

    // rectangular 2x3 @ 3x4
    Tensor r1({2, 3}, 0.0f);
    fillLinear(r1, 0.2f, -0.5f);
    Tensor r2({3, 4}, 0.0f);
    fillLinear(r2, -0.1f, 0.3f);
    MINI_LLAMA_ASSERT_TRUE(expectNear(cudaMatmul(r1, r2), matmul(r1, r2), 1e-4f,
                                      "cudaMatmul rectangular"));

    // 16x16
    Tensor big1({16, 16}, 0.0f);
    fillLinear(big1, 0.01f, -1.0f);
    Tensor big2({16, 16}, 0.0f);
    fillLinear(big2, 0.01f, -0.5f);
    MINI_LLAMA_ASSERT_TRUE(expectNear(
        cudaMatmul(big1, big2), matmul(big1, big2), 1e-4f, "cudaMatmul 16x16"));
    return true;
}

static bool testCudaLinearMatchesCpu() {
    if (skipWithoutCuda("cuda_linear_matches_cpu")) {
        return true;
    }

    Tensor x({4}, 0.0f);
    fillLinear(x, 0.25f, -0.5f);
    Tensor weight({3, 4}, 0.0f);
    fillLinear(weight, 0.1f, -0.2f);
    Tensor bias({3}, 0.0f);
    fillLinear(bias, 0.3f, 0.1f);

    // CPU 参考：linear + bias
    Tensor cpu = linear(x, weight);
    for (int i = 0; i < 3; ++i) {
        cpu[i] += bias[i];
    }

    Tensor gpu = cudaLinear(x, weight, &bias);
    MINI_LLAMA_ASSERT_TRUE(expectNear(gpu, cpu, 1e-4f, "cudaLinear 1D+bias"));

    // 无 bias
    Tensor gpu_no_bias = cudaLinear(x, weight);
    Tensor cpu_no_bias = linear(x, weight);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(gpu_no_bias, cpu_no_bias, 1e-4f, "cudaLinear 1D"));

    // 2D 输入：注意 CPU linearDispatch 只处理单行（忽略 x.shape[0]），
    // 所以参照实现按行调用 CPU linear 再拼成 [2, out_features]。
    Tensor x2({2, 4}, 0.0f);
    fillLinear(x2, -0.15f, 0.7f);
    Tensor cpu2({2, 3}, 0.0f);
    for (int row = 0; row < 2; ++row) {
        Tensor row_in({4}, 0.0f);
        for (int c = 0; c < 4; ++c) {
            row_in[c] = x2.at2(row, c);
        }
        Tensor row_out = linear(row_in, weight);
        for (int c = 0; c < 3; ++c) {
            cpu2.at2(row, c) = row_out[c] + bias[c];
        }
    }
    Tensor gpu2 = cudaLinear(x2, weight, &bias);
    MINI_LLAMA_ASSERT_TRUE(gpu2.shape == std::vector<int>({2, 3}));
    MINI_LLAMA_ASSERT_TRUE(expectNear(gpu2, cpu2, 1e-4f, "cudaLinear 2D"));

    // 设备权重 / 设备输入变体
    CudaDeviceBuffer w_dev(weight.size() * sizeof(float), 0);
    w_dev.upload(weight.data.data(), weight.size() * sizeof(float));

    Tensor gpu_dev_weight =
        cudaLinearDeviceWeight(x, w_dev.data(), weight.shape, &bias);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(gpu_dev_weight, cpu, 1e-4f, "cudaLinearDeviceWeight 1D"));
    Tensor gpu_dev_weight_2d =
        cudaLinearDeviceWeight(x2, w_dev.data(), weight.shape, &bias);
    MINI_LLAMA_ASSERT_TRUE(expectNear(gpu_dev_weight_2d, cpu2, 1e-4f,
                                      "cudaLinearDeviceWeight 2D"));

    CudaTensor x_dev = cudaTensorFromHost(x, 0);
    CudaTensor gpu_dev_input =
        cudaLinearDeviceInput(x_dev, w_dev.data(), weight.shape, 0);
    Tensor dev_input_host = gpu_dev_input.download();
    MINI_LLAMA_ASSERT_TRUE(dev_input_host.shape == std::vector<int>({3}));
    MINI_LLAMA_ASSERT_TRUE(expectNear(dev_input_host, cpu_no_bias, 1e-4f,
                                      "cudaLinearDeviceInput"));
    return true;
}

static bool testCudaMatmulValidation() {
    if (skipWithoutCuda("cuda_matmul_validation")) {
        return true;
    }

    Tensor a({2, 3}, 0.0f);
    Tensor b({2, 3}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([&] { (void)cudaMatmul(a, b); }));

    Tensor a_1d({3}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([&] { (void)cudaMatmul(a_1d, a_1d); }));

    Tensor weight({4, 3}, 0.0f);
    Tensor x({4}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([&] { (void)cudaLinear(x, weight); }));
    Tensor weight_1d({3}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([&] { (void)cudaLinear(x, weight_1d); }));

    CudaDeviceBuffer w_dev(weight.size() * sizeof(float), 0);
    w_dev.upload(weight.data.data(), weight.size() * sizeof(float));
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>([&] {
        (void)cudaLinearDeviceWeight(x, nullptr, weight.shape, nullptr);
    }));
    Tensor bias_bad({2}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>([&] {
        (void)cudaLinearDeviceWeight(x, w_dev.data(), weight.shape, &bias_bad);
    }));
    return true;
}

// ===========================================================================
// quantized linear
// ===========================================================================

static bool testCudaQ80LinearMatchesCpu() {
    if (skipWithoutCuda("cuda_q80_linear_matches_cpu")) {
        return true;
    }

    Tensor weight({4, 32}, 0.0f);
    fillLinear(weight, 0.05f, -1.0f);
    Tensor x({32}, 0.0f);
    fillLinear(x, 0.1f, -0.4f);

    auto blocks = quantizeToQ80(weight);
    Tensor cpu = linearQ80(x, blocks, weight.shape);
    Tensor gpu = cudaQ80Linear(x, blocks, weight.shape);
    MINI_LLAMA_ASSERT_TRUE(expectNear(gpu, cpu, 1e-4f, "cudaQ80Linear"));

    // 2D 输入 + bias
    Tensor x2({2, 32}, 0.0f);
    fillLinear(x2, 0.07f, -0.3f);
    Tensor bias({4}, 0.0f);
    fillLinear(bias, 0.2f, -0.1f);

    Tensor cpu2 = linearQ80(x2, blocks, weight.shape);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 4; ++col) {
            cpu2.at2(row, col) += bias[col];
        }
    }
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(cudaQ80Linear(x2, blocks, weight.shape, &bias), cpu2, 1e-4f,
                   "cudaQ80Linear 2D+bias"));
    return true;
}

static bool testCudaQ40LinearMatchesCpu() {
    if (skipWithoutCuda("cuda_q40_linear_matches_cpu")) {
        return true;
    }

    Tensor weight({4, 32}, 0.0f);
    fillLinear(weight, 0.05f, -1.0f);
    Tensor x({32}, 0.0f);
    fillLinear(x, 0.1f, -0.4f);

    auto blocks = quantizeToQ40(weight);
    Tensor cpu = linearQ40(x, blocks, weight.shape);
    Tensor gpu = cudaQ40Linear(x, blocks, weight.shape);
    MINI_LLAMA_ASSERT_TRUE(expectNear(gpu, cpu, 1e-4f, "cudaQ40Linear"));
    return true;
}

static bool testCudaQ41LinearMatchesCpu() {
    if (skipWithoutCuda("cuda_q41_linear_matches_cpu")) {
        return true;
    }

    Tensor x({32}, 0.0f);
    fillLinear(x, 0.1f, -0.4f);

    auto blocks = makeQ41Blocks(4, 32);
    std::vector<int> shape = {4, 32};
    Tensor cpu = linearQ41(x, blocks, shape);
    Tensor gpu = cudaQ41Linear(x, blocks, shape);
    MINI_LLAMA_ASSERT_TRUE(expectNear(gpu, cpu, 1e-4f, "cudaQ41Linear"));
    return true;
}

static bool testCudaQuantDeviceVariants() {
    if (skipWithoutCuda("cuda_quant_device_variants")) {
        return true;
    }

    Tensor weight({4, 32}, 0.0f);
    fillLinear(weight, 0.05f, -1.0f);
    Tensor x({32}, 0.0f);
    fillLinear(x, 0.1f, -0.4f);

    auto blocks = quantizeToQ80(weight);
    Tensor host = cudaQ80Linear(x, blocks, weight.shape);

    CudaDeviceBuffer w_dev(blocks.size() * sizeof(BlockQ80), 0);
    w_dev.upload(blocks.data(), blocks.size() * sizeof(BlockQ80));

    Tensor dev_weight = cudaQ80LinearDeviceWeight(
        x, w_dev.data(), blocks.size(), weight.shape, nullptr);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(dev_weight, host, 1e-5f, "cudaQ80LinearDeviceWeight"));

    CudaTensor x_dev = cudaTensorFromHost(x, 0);
    CudaTensor dev_input = cudaQ80LinearDeviceInput(
        x_dev, w_dev.data(), blocks.size(), weight.shape, 0);
    Tensor dev_input_host = dev_input.download();
    MINI_LLAMA_ASSERT_TRUE(dev_input_host.shape == std::vector<int>({4}));
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(dev_input_host, host, 1e-5f, "cudaQ80LinearDeviceInput"));
    return true;
}

static bool testCudaQuantValidation() {
    if (skipWithoutCuda("cuda_quant_validation")) {
        return true;
    }

    Tensor weight({4, 32}, 0.0f);
    fillLinear(weight, 0.05f, -1.0f);
    auto blocks = quantizeToQ80(weight);

    // in_features 不匹配
    Tensor x_bad({16}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { (void)cudaQ80Linear(x_bad, blocks, weight.shape); }));

    // block 数量不匹配
    auto short_blocks = blocks;
    short_blocks.pop_back();
    Tensor x({32}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { (void)cudaQ80Linear(x, short_blocks, weight.shape); }));

    // 空权重指针
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>([&] {
        (void)cudaQ80LinearDeviceWeight(x, nullptr, blocks.size(), weight.shape,
                                        nullptr);
    }));

    // 输入 rank 非法
    Tensor x_3d({1, 1, 32}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { (void)cudaQ80Linear(x_3d, blocks, weight.shape); }));
    return true;
}

// ===========================================================================
// KV cache
// ===========================================================================

static bool testCudaKvCacheRoundtrip() {
    if (skipWithoutCuda("cuda_kv_cache_roundtrip")) {
        return true;
    }

    CudaKvCache cache(2, 4, 2, 3, 0);
    MINI_LLAMA_ASSERT_EQ(cache.nLayers(), 2);
    MINI_LLAMA_ASSERT_EQ(cache.maxSeqLen(), 4);
    MINI_LLAMA_ASSERT_EQ(cache.nKvHeads(), 2);
    MINI_LLAMA_ASSERT_EQ(cache.headDim(), 3);
    MINI_LLAMA_ASSERT_TRUE(!cache.empty());
    MINI_LLAMA_ASSERT_EQ(
        cache.bytes(), static_cast<size_t>(2) * 4 * 2 * 3 * sizeof(float) * 2);

    Tensor k({2, 3}, 0.0f);
    Tensor v({2, 3}, 0.0f);
    fillLinear(k, 0.3f, 1.0f);
    fillLinear(v, -0.2f, 5.0f);

    cache.write(1, 2, k, v);

    Tensor key = cache.readKey(1, 2);
    MINI_LLAMA_ASSERT_TRUE(key.shape == std::vector<int>({2, 3}));
    MINI_LLAMA_ASSERT_TRUE(expectNear(key, k, 1e-6f, "readKey"));

    Tensor value = cache.readValue(1, 2);
    MINI_LLAMA_ASSERT_TRUE(expectNear(value, v, 1e-6f, "readValue"));

    Tensor key_head = cache.readKeyHead(1, 2, 1);
    MINI_LLAMA_ASSERT_TRUE(key_head.shape == std::vector<int>({3}));
    for (int d = 0; d < 3; ++d) {
        MINI_LLAMA_ASSERT_NEAR(key_head[d], k.at2(1, d), 1e-6f);
    }
    Tensor value_head = cache.readValueHead(1, 2, 0);
    for (int d = 0; d < 3; ++d) {
        MINI_LLAMA_ASSERT_NEAR(value_head[d], v.at2(0, d), 1e-6f);
    }

    // 未写入的位置保持 0
    Tensor untouched = cache.readKey(0, 0);
    for (size_t i = 0; i < untouched.size(); ++i) {
        MINI_LLAMA_ASSERT_NEAR(untouched.data[i], 0.0f, 1e-6f);
    }
    return true;
}

static bool testCudaKvCacheWriteDevice() {
    if (skipWithoutCuda("cuda_kv_cache_write_device")) {
        return true;
    }

    CudaKvCache cache(1, 4, 2, 3, 0);
    Tensor k({2, 3}, 0.0f);
    Tensor v({2, 3}, 0.0f);
    fillLinear(k, 0.4f, -2.0f);
    fillLinear(v, 0.25f, 0.5f);

    CudaTensor k_dev = cudaTensorFromHost(k, 0);
    CudaTensor v_dev = cudaTensorFromHost(v, 0);
    cache.writeDevice(0, 3, k_dev, v_dev);

    MINI_LLAMA_ASSERT_TRUE(
        expectNear(cache.readKey(0, 3), k, 1e-6f, "writeDevice key"));
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(cache.readValue(0, 3), v, 1e-6f, "writeDevice value"));
    return true;
}

static bool testCudaKvCacheClearAndMove() {
    if (skipWithoutCuda("cuda_kv_cache_clear_and_move")) {
        return true;
    }

    CudaKvCache cache(1, 4, 2, 3, 0);
    Tensor k({2, 3}, 7.0f);
    Tensor v({2, 3}, 8.0f);
    cache.write(0, 1, k, v);

    cache.clear();
    Tensor key = cache.readKey(0, 1);
    for (size_t i = 0; i < key.size(); ++i) {
        MINI_LLAMA_ASSERT_NEAR(key.data[i], 0.0f, 1e-6f);
    }

    // 移动后目标可用，源变空
    CudaKvCache moved(std::move(cache));
    MINI_LLAMA_ASSERT_EQ(moved.nLayers(), 1);
    MINI_LLAMA_ASSERT_TRUE(!moved.empty());
    MINI_LLAMA_ASSERT_TRUE(cache.empty());

    Tensor k2({2, 3}, 1.5f);
    moved.write(0, 2, k2, v);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(moved.readKey(0, 2), k2, 1e-6f, "moved cache write/read"));
    return true;
}

static bool testCudaKvCacheValidation() {
    if (skipWithoutCuda("cuda_kv_cache_validation")) {
        return true;
    }

    CudaKvCache cache(1, 2, 2, 3, 0);
    Tensor k({2, 3}, 1.0f);
    Tensor v({2, 3}, 2.0f);

    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::out_of_range>([&] { cache.write(1, 0, k, v); }));
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::out_of_range>([&] { cache.write(0, 2, k, v); }));
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::out_of_range>([&] { cache.write(0, -1, k, v); }));
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::out_of_range>([&] { (void)cache.readKeyHead(0, 0, 2); }));
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::out_of_range>(
        [&] { (void)cache.readValueHead(0, 0, -1); }));

    Tensor wrong_heads({1, 3}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::out_of_range>(
        [&] { cache.write(0, 0, wrong_heads, v); }));
    Tensor wrong_dim({2, 4}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::out_of_range>([&] { cache.write(0, 0, k, wrong_dim); }));

    // 空 cache 的所有访问都应报错
    CudaKvCache empty_cache;
    MINI_LLAMA_ASSERT_TRUE(empty_cache.empty());
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([&] { (void)empty_cache.readKey(0, 0); }));
    MINI_LLAMA_ASSERT_TRUE(
        throwsAs<std::runtime_error>([&] { (void)empty_cache.keysData(); }));

    // reset 重新分配后可正常使用
    cache.reset(2, 8, 2, 4, 0);
    MINI_LLAMA_ASSERT_EQ(cache.maxSeqLen(), 8);
    MINI_LLAMA_ASSERT_EQ(cache.headDim(), 4);
    Tensor k2({2, 4}, 3.0f);
    Tensor v2({2, 4}, 4.0f);
    cache.write(1, 7, k2, v2);
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(cache.readKey(1, 7), k2, 1e-6f, "reset 后读写"));
    return true;
}

// ===========================================================================
// attention
// ===========================================================================

static bool testCudaAttentionMatchesReference() {
    if (skipWithoutCuda("cuda_attention_matches_reference")) {
        return true;
    }

    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 4;
    CudaKvCache cache(1, 8, n_kv_heads, head_dim, 0);

    // 写入 pos 0..3 的确定性 K/V
    for (int pos = 0; pos <= 3; ++pos) {
        Tensor k({n_kv_heads, head_dim}, 0.0f);
        Tensor v({n_kv_heads, head_dim}, 0.0f);
        for (size_t i = 0; i < k.size(); ++i) {
            k.data[i] = 0.1f * static_cast<float>(pos + 1) +
                        0.03f * static_cast<float>(i) - 0.2f;
            v.data[i] = -0.05f * static_cast<float>(pos) +
                        0.07f * static_cast<float>(i) + 0.4f;
        }
        cache.write(0, pos, k, v);
    }

    Tensor q({n_heads, head_dim}, 0.0f);
    fillLinear(q, 0.15f, -0.3f);

    const int pos = 3;
    Tensor gpu =
        cudaAttentionDecode(q, cache, 0, pos, n_heads, n_kv_heads, head_dim, 0);
    Tensor cpu =
        attentionReference(q, cache, 0, pos, n_heads, n_kv_heads, head_dim);

    MINI_LLAMA_ASSERT_TRUE(gpu.shape == std::vector<int>({n_heads, head_dim}));
    MINI_LLAMA_ASSERT_TRUE(expectNear(gpu, cpu, 1e-4f, "cudaAttentionDecode"));
    return true;
}

static bool testCudaAttentionPos0ReturnsV() {
    if (skipWithoutCuda("cuda_attention_pos0_returns_v")) {
        return true;
    }

    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 4;
    CudaKvCache cache(1, 4, n_kv_heads, head_dim, 0);

    Tensor k({n_kv_heads, head_dim}, 0.0f);
    Tensor v({n_kv_heads, head_dim}, 0.0f);
    fillLinear(k, 0.2f, -1.0f);
    fillLinear(v, 0.5f, 10.0f);
    cache.write(0, 0, k, v);

    Tensor q({n_heads, head_dim}, 0.0f);
    fillLinear(q, 0.3f, 0.25f);

    Tensor out =
        cudaAttentionDecode(q, cache, 0, 0, n_heads, n_kv_heads, head_dim, 0);
    // pos=0 时 softmax 权重恒为 1，输出应等于对应 KV head 的 V
    const int group = n_heads / n_kv_heads;
    for (int h = 0; h < n_heads; ++h) {
        Tensor v_head = cache.readValueHead(0, 0, h / group);
        for (int d = 0; d < head_dim; ++d) {
            MINI_LLAMA_ASSERT_NEAR(out.at2(h, d), v_head[d], 1e-5f);
        }
    }
    return true;
}

static bool testCudaAttentionGqaMapping() {
    if (skipWithoutCuda("cuda_attention_gqa_mapping")) {
        return true;
    }

    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 4;
    CudaKvCache cache(1, 4, n_kv_heads, head_dim, 0);

    // K 全 0 => score 全 0 => 均匀权重；V[kv0]=1, V[kv1]=2
    Tensor k({n_kv_heads, head_dim}, 0.0f);
    Tensor v({n_kv_heads, head_dim}, 0.0f);
    for (int d = 0; d < head_dim; ++d) {
        v.at2(0, d) = 1.0f;
        v.at2(1, d) = 2.0f;
    }
    cache.write(0, 0, k, v);
    cache.write(0, 1, k, v);
    cache.write(0, 2, k, v);

    Tensor q({n_heads, head_dim}, 0.0f);
    fillLinear(q, 0.1f, 0.0f);

    Tensor out =
        cudaAttentionDecode(q, cache, 0, 2, n_heads, n_kv_heads, head_dim, 0);
    for (int d = 0; d < head_dim; ++d) {
        MINI_LLAMA_ASSERT_NEAR(out.at2(0, d), 1.0f, 1e-5f);
        MINI_LLAMA_ASSERT_NEAR(out.at2(1, d), 1.0f, 1e-5f);
        MINI_LLAMA_ASSERT_NEAR(out.at2(2, d), 2.0f, 1e-5f);
        MINI_LLAMA_ASSERT_NEAR(out.at2(3, d), 2.0f, 1e-5f);
    }
    return true;
}

static bool testCudaAttentionDeviceInput() {
    if (skipWithoutCuda("cuda_attention_device_input")) {
        return true;
    }

    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 4;
    CudaKvCache cache(1, 4, n_kv_heads, head_dim, 0);

    for (int pos = 0; pos <= 2; ++pos) {
        Tensor k({n_kv_heads, head_dim}, 0.0f);
        Tensor v({n_kv_heads, head_dim}, 0.0f);
        fillLinear(k, 0.1f * (pos + 1), -0.5f);
        fillLinear(v, -0.1f, 0.9f + 0.2f * pos);
        cache.write(0, pos, k, v);
    }

    Tensor q({n_heads, head_dim}, 0.0f);
    fillLinear(q, 0.12f, -0.4f);

    Tensor host =
        cudaAttentionDecode(q, cache, 0, 2, n_heads, n_kv_heads, head_dim, 0);

    CudaTensor q_dev = cudaTensorFromHost(q, 0);
    CudaTensor dev = cudaAttentionDecodeDeviceInput(q_dev, cache, 0, 2, n_heads,
                                                    n_kv_heads, head_dim, 0);
    Tensor dev_host = dev.download();

    // 设备变体返回扁平化 1D [n_heads * head_dim]，host 变体为 2D
    MINI_LLAMA_ASSERT_TRUE(dev_host.shape ==
                           std::vector<int>({n_heads * head_dim}));
    Tensor host_flat = host.reshapeChecked({n_heads * head_dim},
                                           "attentionDeviceInput reference");
    MINI_LLAMA_ASSERT_TRUE(
        expectNear(dev_host, host_flat, 1e-4f, "attentionDeviceInput"));
    return true;
}

static bool testCudaAttentionMultiLayerPosition() {
    if (skipWithoutCuda("cuda_attention_multi_layer_position")) {
        return true;
    }

    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 4;
    CudaKvCache cache(2, 8, n_kv_heads, head_dim, 0);

    // layer 0 与 layer 1 写不同数据，pos 到 5；只对 layer 1 / pos 5 做对照
    for (int layer = 0; layer < 2; ++layer) {
        for (int pos = 0; pos <= 5; ++pos) {
            Tensor k({n_kv_heads, head_dim}, 0.0f);
            Tensor v({n_kv_heads, head_dim}, 0.0f);
            for (size_t i = 0; i < k.size(); ++i) {
                k.data[i] = 0.02f * static_cast<float>(i) +
                            0.11f * static_cast<float>(layer) +
                            0.05f * static_cast<float>(pos) - 0.6f;
                v.data[i] = -0.03f * static_cast<float>(i) +
                            0.07f * static_cast<float>(layer) -
                            0.04f * static_cast<float>(pos) + 0.8f;
            }
            cache.write(layer, pos, k, v);
        }
    }

    Tensor q({n_heads, head_dim}, 0.0f);
    fillLinear(q, -0.09f, 0.35f);

    Tensor gpu =
        cudaAttentionDecode(q, cache, 1, 5, n_heads, n_kv_heads, head_dim, 0);
    Tensor cpu =
        attentionReference(q, cache, 1, 5, n_heads, n_kv_heads, head_dim);
    MINI_LLAMA_ASSERT_TRUE(expectNear(gpu, cpu, 1e-4f, "attention layer/pos"));
    return true;
}

static bool testCudaAttentionValidation() {
    if (skipWithoutCuda("cuda_attention_validation")) {
        return true;
    }

    const int head_dim = 4;
    CudaKvCache cache(1, 8, 2, head_dim, 0);
    Tensor k({2, head_dim}, 0.0f);
    Tensor v({2, head_dim}, 0.0f);
    cache.write(0, 0, k, v);
    Tensor q({4, head_dim}, 0.0f);

    // 空 cache
    CudaKvCache empty_cache;
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>([&] {
        (void)cudaAttentionDecode(q, empty_cache, 0, 0, 4, 2, head_dim, 0);
    }));

    // q 形状错误
    Tensor q_bad({4, 5}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>([&] {
        (void)cudaAttentionDecode(q_bad, cache, 0, 0, 4, 2, head_dim, 0);
    }));

    // layer / pos 越界
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::out_of_range>(
        [&] { (void)cudaAttentionDecode(q, cache, 1, 0, 4, 2, head_dim, 0); }));
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::out_of_range>(
        [&] { (void)cudaAttentionDecode(q, cache, 0, 8, 4, 2, head_dim, 0); }));

    // n_heads 不能被 n_kv_heads 整除
    Tensor q3({3, head_dim}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>([&] {
        (void)cudaAttentionDecode(q3, cache, 0, 0, 3, 2, head_dim, 0);
    }));

    // cache 的 n_kv_heads/head_dim 与入参不符
    MINI_LLAMA_ASSERT_TRUE(throwsAs<std::runtime_error>(
        [&] { (void)cudaAttentionDecode(q, cache, 0, 0, 4, 2, 8, 0); }));
    return true;
}

// ===========================================================================
// 端到端：CPU vs CUDA
// ===========================================================================

static bool testCudaForwardMatchesCpuLogits() {
    if (skipWithoutCuda("cuda_forward_matches_cpu_logits")) {
        return true;
    }

    MiniLlamaModel cpu_model =
        loadModel("models/tiny/model.json", "models/tiny/model.bin");
    MiniLlamaModel cuda_model =
        loadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (!cpu_model.loaded || !cuda_model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load tiny model");
    }

    uploadModelWeightsToCuda(cuda_model, 0);
    MINI_LLAMA_ASSERT_TRUE(modelHasCudaWeights(cuda_model));
    MINI_LLAMA_ASSERT_EQ(
        static_cast<int>(modelCudaUploadedWeightCount(cuda_model)), 21);
    MINI_LLAMA_ASSERT_TRUE(modelCudaMemoryBytes(cuda_model) > 0);

    MiniLlamaContext cpu_ctx(&cpu_model);
    MiniLlamaContext cuda_ctx(&cuda_model);
    MINI_LLAMA_ASSERT_TRUE(!cuda_ctx.cuda_kv_cache.empty());

    AsciiTokenizer tokenizer;
    std::vector<int> tokens = tokenizer.encode("hello");

    float worst_diff = 0.0f;
    for (size_t i = 0; i < tokens.size(); ++i) {
        cpu_ctx.pos = static_cast<int>(i);
        cuda_ctx.pos = static_cast<int>(i);

        Tensor logits_cpu = forwardToken(cpu_ctx, cpu_model, tokens[i]);
        Tensor logits_cuda = forwardToken(cuda_ctx, cuda_model, tokens[i]);

        MINI_LLAMA_ASSERT_EQ(logits_cpu.numDims(), 1);
        MINI_LLAMA_ASSERT_EQ(logits_cuda.numDims(), 1);
        MINI_LLAMA_ASSERT_EQ(logits_cpu.shape[0], cpu_model.config.vocab_size);
        MINI_LLAMA_ASSERT_EQ(logits_cuda.shape[0],
                             cuda_model.config.vocab_size);
        MINI_LLAMA_ASSERT_TRUE(
            expectNear(logits_cuda, logits_cpu, 5e-3f, "CPU/CUDA logits"));
        worst_diff = std::max(worst_diff, maxAbsDiff(logits_cuda, logits_cpu));
    }
    std::cout << "  (max logits diff over prefill: " << worst_diff << ")\n";

    // CUDA 路径必须真的被使用：attention 走内核且无 CPU 回退
    MINI_LLAMA_ASSERT_TRUE(cuda_model.cuda_weights->attention_calls > 0);
    MINI_LLAMA_ASSERT_EQ(cuda_model.cuda_weights->attention_cpu_fallbacks,
                         static_cast<size_t>(0));
    MINI_LLAMA_ASSERT_TRUE(cuda_model.cuda_weights->linear_calls > 0);
    MINI_LLAMA_ASSERT_TRUE(cuda_model.cuda_weights->kv_cache_write_bytes > 0);

    // GPU/CPU KV cache 内容一致
    const int n_heads = cpu_model.config.n_kv_heads;
    const int head_dim = cpu_model.config.head_dim;
    for (int layer = 0; layer < cpu_model.config.n_layers; ++layer) {
        for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) {
            for (int head = 0; head < n_heads; ++head) {
                Tensor gpu_key =
                    cuda_ctx.cuda_kv_cache.readKeyHead(layer, pos, head);
                const float* cpu_key =
                    cpu_ctx.kv_cache.keyPtr(layer, pos, head);
                for (int d = 0; d < head_dim; ++d) {
                    MINI_LLAMA_ASSERT_NEAR(gpu_key[d], cpu_key[d], 5e-3f);
                }

                Tensor gpu_val =
                    cuda_ctx.cuda_kv_cache.readValueHead(layer, pos, head);
                const float* cpu_val =
                    cpu_ctx.kv_cache.valuePtr(layer, pos, head);
                for (int d = 0; d < head_dim; ++d) {
                    MINI_LLAMA_ASSERT_NEAR(gpu_val[d], cpu_val[d], 5e-3f);
                }
            }
        }
    }
    return true;
}

static bool testCudaForwardGreedyGenerationMatchesCpu() {
    if (skipWithoutCuda("cuda_forward_greedy_generation_matches_cpu")) {
        return true;
    }

    MiniLlamaModel cpu_model =
        loadModel("models/tiny/model.json", "models/tiny/model.bin");
    MiniLlamaModel cuda_model =
        loadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (!cpu_model.loaded || !cuda_model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load tiny model");
    }
    uploadModelWeightsToCuda(cuda_model, 0);

    MiniLlamaContext cpu_ctx(&cpu_model);
    MiniLlamaContext cuda_ctx(&cuda_model);

    AsciiTokenizer tokenizer;
    std::vector<int> prompt = tokenizer.encode("hi");
    const int n_decode = 4;

    // prefill 后逐 token 贪心解码，比较每一步的 argmax
    int pos = 0;
    int next_cpu = -1;
    int next_cuda = -1;
    for (size_t i = 0; i < prompt.size(); ++i, ++pos) {
        cpu_ctx.pos = pos;
        cuda_ctx.pos = pos;

        Tensor logits_cpu = forwardToken(cpu_ctx, cpu_model, prompt[i]);
        Tensor logits_cuda = forwardToken(cuda_ctx, cuda_model, prompt[i]);
        next_cpu = MiniSampler::sampleGreedy(logits_cpu);
        next_cuda = MiniSampler::sampleGreedy(logits_cuda);
        MINI_LLAMA_ASSERT_EQ(next_cpu, next_cuda);
    }

    for (int step = 0; step < n_decode; ++step, ++pos) {
        cpu_ctx.pos = pos;
        cuda_ctx.pos = pos;

        Tensor logits_cpu = forwardToken(cpu_ctx, cpu_model, next_cpu);
        Tensor logits_cuda = forwardToken(cuda_ctx, cuda_model, next_cuda);
        next_cpu = MiniSampler::sampleGreedy(logits_cpu);
        next_cuda = MiniSampler::sampleGreedy(logits_cuda);
        MINI_LLAMA_ASSERT_EQ(next_cpu, next_cuda);
    }

    // 解码结束后 KV cache 仍逐步一致（覆盖 decode 阶段 attention）
    for (int layer = 0; layer < cuda_model.config.n_layers; ++layer) {
        for (int p = 0; p <= pos; ++p) {
            Tensor gpu_key = cuda_ctx.cuda_kv_cache.readKey(layer, p);
            for (int head = 0; head < cuda_model.config.n_kv_heads; ++head) {
                const float* cpu_key = cpu_ctx.kv_cache.keyPtr(layer, p, head);
                for (int d = 0; d < cuda_model.config.head_dim; ++d) {
                    MINI_LLAMA_ASSERT_NEAR(gpu_key.at2(head, d), cpu_key[d],
                                           5e-3f);
                }
            }
        }
    }
    return true;
}

// ===========================================================================
// 注册
// ===========================================================================

static struct CudaTestRegistrar {
    CudaTestRegistrar() {
        registerTest("cuda_runtime_built_flag", testCudaRuntimeBuiltFlag);
        registerTest("cuda_device_info", testCudaDeviceInfo);
        registerTest("cuda_device_buffer_roundtrip",
                     testCudaDeviceBufferRoundtrip);
        registerTest("cuda_tensor_roundtrip", testCudaTensorRoundtrip);
        registerTest("cuda_rms_norm_matches_cpu", testCudaRmsNormMatchesCpu);
        registerTest("cuda_silu_matches_cpu", testCudaSiluMatchesCpu);
        registerTest("cuda_elementwise_matches_cpu",
                     testCudaElementwiseMatchesCpu);
        registerTest("cuda_softmax_matches_cpu", testCudaSoftmaxMatchesCpu);
        registerTest("cuda_ops_device_variants_match_host",
                     testCudaOpsDeviceVariantsMatchHost);
        registerTest("cuda_embedding_lookup", testCudaEmbeddingLookup);
        registerTest("cuda_rope_matches_cpu", testCudaRopeMatchesCpu);
        registerTest("cuda_matmul_matches_cpu", testCudaMatmulMatchesCpu);
        registerTest("cuda_linear_matches_cpu", testCudaLinearMatchesCpu);
        registerTest("cuda_matmul_validation", testCudaMatmulValidation);
        registerTest("cuda_q80_linear_matches_cpu",
                     testCudaQ80LinearMatchesCpu);
        registerTest("cuda_q40_linear_matches_cpu",
                     testCudaQ40LinearMatchesCpu);
        registerTest("cuda_q41_linear_matches_cpu",
                     testCudaQ41LinearMatchesCpu);
        registerTest("cuda_quant_device_variants", testCudaQuantDeviceVariants);
        registerTest("cuda_quant_validation", testCudaQuantValidation);
        registerTest("cuda_kv_cache_roundtrip", testCudaKvCacheRoundtrip);
        registerTest("cuda_kv_cache_write_device", testCudaKvCacheWriteDevice);
        registerTest("cuda_kv_cache_clear_and_move",
                     testCudaKvCacheClearAndMove);
        registerTest("cuda_kv_cache_validation", testCudaKvCacheValidation);
        registerTest("cuda_attention_matches_reference",
                     testCudaAttentionMatchesReference);
        registerTest("cuda_attention_pos0_returns_v",
                     testCudaAttentionPos0ReturnsV);
        registerTest("cuda_attention_gqa_mapping", testCudaAttentionGqaMapping);
        registerTest("cuda_attention_device_input",
                     testCudaAttentionDeviceInput);
        registerTest("cuda_attention_multi_layer_position",
                     testCudaAttentionMultiLayerPosition);
        registerTest("cuda_attention_validation", testCudaAttentionValidation);
        registerTest("cuda_forward_matches_cpu_logits",
                     testCudaForwardMatchesCpuLogits);
        registerTest("cuda_forward_greedy_generation_matches_cpu",
                     testCudaForwardGreedyGenerationMatchesCpu);
    }
} cuda_test_registrar;
