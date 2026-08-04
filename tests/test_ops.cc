// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/ops.h"
#include "mini_llama/tensor.h"
#include "tests/test_main.h"

// ==========================================================================
// mini_llama::matmul
// ==========================================================================
static bool testMatmulIdentity() {
    mini_llama::Tensor a({2, 2}, 0.0f);
    a[0] = 1.0f;
    a[1] = 0.0f;
    a[2] = 0.0f;
    a[3] = 1.0f;
    mini_llama::Tensor b({2, 2}, 0.0f);
    b[0] = 1.0f;
    b[1] = 2.0f;
    b[2] = 3.0f;
    b[3] = 4.0f;
    mini_llama::Tensor c = mini_llama::matmul(a, b);
    MINI_LLAMA_ASSERT_EQ(c.shape[0], 2);
    MINI_LLAMA_ASSERT_EQ(c.shape[1], 2);
    MINI_LLAMA_ASSERT_NEAR(c[0], 1.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(c[1], 2.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(c[2], 3.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(c[3], 4.0f, 1e-5f);
    return true;
}

static bool testMatmulSimple() {
    mini_llama::Tensor a({2, 2}, 0.0f);
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    a[3] = 4.0f;
    mini_llama::Tensor b({2, 2}, 0.0f);
    b[0] = 5.0f;
    b[1] = 6.0f;
    b[2] = 7.0f;
    b[3] = 8.0f;
    mini_llama::Tensor c = mini_llama::matmul(a, b);
    MINI_LLAMA_ASSERT_NEAR(c[0], 19.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(c[1], 22.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(c[2], 43.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(c[3], 50.0f, 1e-5f);
    return true;
}

static bool testMatmulRectangular() {
    // a: [2, 3], b: [3, 4] -> c: [2, 4]
    mini_llama::Tensor a({2, 3}, 0.0f);
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    a[3] = 4.0f;
    a[4] = 5.0f;
    a[5] = 6.0f;
    mini_llama::Tensor b({3, 4}, 0.0f);
    for (int i = 0; i < 12; ++i) {
        b[i] = static_cast<float>(i);
    }
    mini_llama::Tensor c = mini_llama::matmul(a, b);
    MINI_LLAMA_ASSERT_EQ(c.shape[0], 2);
    MINI_LLAMA_ASSERT_EQ(c.shape[1], 4);
    // c[0,0] = 1*0 + 2*4 + 3*8 = 0 + 8 + 24 = 32
    MINI_LLAMA_ASSERT_NEAR(c.at2(0, 0), 32.0f, 1e-5f);
    // c[1,0] = 4*0 + 5*4 + 6*8 = 0 + 20 + 48 = 68
    MINI_LLAMA_ASSERT_NEAR(c.at2(1, 0), 68.0f, 1e-5f);
    return true;
}

static bool testMatmulShapeMismatch() {
    mini_llama::Tensor a({2, 3}, 0.0f);
    mini_llama::Tensor b({2, 3}, 0.0f);
    try {
        mini_llama::matmul(a, b);
        MINI_LLAMA_ASSERT_FAIL(
            "expected exception for mini_llama::matmul shape mismatch");
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

// ==========================================================================
// Linear
// ==========================================================================
static bool testLinear1d() {
    mini_llama::Tensor x({3}, 0.0f);
    x[0] = 1.0f;
    x[1] = 2.0f;
    x[2] = 3.0f;
    mini_llama::Tensor weight({3, 3}, 0.0f);
    weight[0] = 1.0f;
    weight[4] = 1.0f;
    weight[8] = 1.0f;
    mini_llama::Tensor y = mini_llama::linear(x, weight);
    MINI_LLAMA_ASSERT_EQ(y.numDims(), 1);
    MINI_LLAMA_ASSERT_EQ(y.shape[0], 3);
    MINI_LLAMA_ASSERT_NEAR(y[0], 1.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(y[1], 2.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(y[2], 3.0f, 1e-5f);
    return true;
}

static bool testLinear2d() {
    mini_llama::Tensor x({1, 3}, 0.0f);
    x[0] = 1.0f;
    x[1] = 2.0f;
    x[2] = 3.0f;
    mini_llama::Tensor weight({2, 3}, 0.0f);
    weight[0] = 1.0f;
    weight[1] = 0.0f;
    weight[2] = 0.0f;
    weight[3] = 0.0f;
    weight[4] = 1.0f;
    weight[5] = 0.0f;
    mini_llama::Tensor y = mini_llama::linear(x, weight);
    MINI_LLAMA_ASSERT_EQ(y.numDims(), 2);
    MINI_LLAMA_ASSERT_EQ(y.shape[0], 1);
    MINI_LLAMA_ASSERT_EQ(y.shape[1], 2);
    MINI_LLAMA_ASSERT_NEAR(y[0], 1.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(y[1], 2.0f, 1e-5f);
    return true;
}

static bool testLinearShapeMismatch() {
    mini_llama::Tensor x({3}, 0.0f);
    mini_llama::Tensor weight({2, 4}, 0.0f);
    try {
        mini_llama::linear(x, weight);
        MINI_LLAMA_ASSERT_FAIL("expected exception for linear shape mismatch");
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

// ==========================================================================
// RmsNorm
// ==========================================================================
static bool testRmsNormUnit() {
    mini_llama::Tensor x({4}, 1.0f);
    mini_llama::Tensor w({4}, 1.0f);
    mini_llama::Tensor y = mini_llama::rmsNorm(x, w, 1e-5f);
    MINI_LLAMA_ASSERT_EQ(y.shape[0], 4);
    for (int i = 0; i < 4; ++i) {
        MINI_LLAMA_ASSERT_NEAR(y[i], 1.0f, 1e-4f);
    }
    return true;
}

static bool testRmsNormScaled() {
    mini_llama::Tensor x({4}, 0.0f);
    x[0] = 2.0f;
    mini_llama::Tensor w({4}, 1.0f);
    mini_llama::Tensor y = mini_llama::rmsNorm(x, w, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(y[0], 2.0f, 1e-4f);
    MINI_LLAMA_ASSERT_NEAR(y[1], 0.0f, 1e-4f);
    return true;
}

static bool testRmsNormWithWeight() {
    // x = [1, 1, 1, 1], weight = [2, 2, 2, 2]
    // rms = 1, scale = 1, y = x * scale * weight = [2, 2, 2, 2]
    mini_llama::Tensor x({4}, 1.0f);
    mini_llama::Tensor w({4}, 2.0f);
    mini_llama::Tensor y = mini_llama::rmsNorm(x, w, 1e-5f);
    for (int i = 0; i < 4; ++i) {
        MINI_LLAMA_ASSERT_NEAR(y[i], 2.0f, 1e-4f);
    }
    return true;
}

static bool testRmsNormShapeMismatch() {
    mini_llama::Tensor x({4}, 0.0f);
    mini_llama::Tensor w({3}, 0.0f);
    try {
        mini_llama::rmsNorm(x, w, 1e-5f);
        MINI_LLAMA_ASSERT_FAIL("expected exception for rmsNorm shape mismatch");
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

// ==========================================================================
// Softmax
// ==========================================================================
static bool testSoftmax() {
    mini_llama::Tensor x({3}, 0.0f);
    x[0] = 1.0f;
    x[1] = 2.0f;
    x[2] = 3.0f;
    mini_llama::Tensor y = mini_llama::softmax(x);
    MINI_LLAMA_ASSERT_EQ(y.shape[0], 3);
    MINI_LLAMA_ASSERT_NEAR(y[0], 0.09003057f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(y[1], 0.24472847f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(y[2], 0.66524096f, 1e-5f);
    float sum = y[0] + y[1] + y[2];
    MINI_LLAMA_ASSERT_NEAR(sum, 1.0f, 1e-5f);
    return true;
}

static bool testSoftmaxUniform() {
    // Softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
    mini_llama::Tensor x({3}, 0.0f);
    mini_llama::Tensor y = mini_llama::softmax(x);
    for (int i = 0; i < 3; ++i) {
        MINI_LLAMA_ASSERT_NEAR(y[i], 1.0f / 3.0f, 1e-5f);
    }
    return true;
}

static bool testSoftmaxWrongDim() {
    mini_llama::Tensor x({2, 3}, 0.0f);
    try {
        mini_llama::softmax(x);
        MINI_LLAMA_ASSERT_FAIL(
            "expected exception for softmax on 2D mini_llama::Tensor");
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

static bool testSoftmaxEmpty() {
    mini_llama::Tensor x;
    x.shape = {0};
    try {
        mini_llama::softmax(x);
        MINI_LLAMA_ASSERT_FAIL(
            "expected exception for softmax on empty mini_llama::Tensor");
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

// ==========================================================================
// Silu
// ==========================================================================
static bool testSilu() {
    mini_llama::Tensor x({1}, 0.0f);
    mini_llama::Tensor y = mini_llama::silu(x);
    MINI_LLAMA_ASSERT_NEAR(y[0], 0.0f, 1e-5f);

    x[0] = 1.0f;
    y = mini_llama::silu(x);
    MINI_LLAMA_ASSERT_NEAR(y[0], 0.7310586f, 1e-5f);
    return true;
}

// ==========================================================================
// SwiGlu
// ==========================================================================
static bool testSwiglu() {
    // SwiGlu(gate, up) = Silu(gate) * up
    mini_llama::Tensor gate({3}, 0.0f);
    gate[0] = 0.0f;
    gate[1] = 1.0f;
    gate[2] = 2.0f;
    mini_llama::Tensor up({3}, 0.0f);
    up[0] = 1.0f;
    up[1] = 2.0f;
    up[2] = 3.0f;
    mini_llama::Tensor y = mini_llama::swiGlu(gate, up);

    // Silu(0) = 0, so SwiGlu(0, 1) = 0
    MINI_LLAMA_ASSERT_NEAR(y[0], 0.0f, 1e-5f);
    // Silu(1) ≈ 0.731, so SwiGlu(1, 2) ≈ 0.731 * 2 ≈ 1.462
    MINI_LLAMA_ASSERT_NEAR(y[1], 0.7310586f * 2.0f, 1e-5f);
    // Silu(2) ≈ 1.761, so SwiGlu(2, 3) ≈ 1.761 * 3 ≈ 5.284
    MINI_LLAMA_ASSERT_NEAR(y[2], 1.7615942f * 3.0f, 1e-4f);
    return true;
}

static bool testSwigluShapeMismatch() {
    mini_llama::Tensor a({3}, 0.0f);
    mini_llama::Tensor b({4}, 0.0f);
    try {
        mini_llama::swiGlu(a, b);
        MINI_LLAMA_ASSERT_FAIL("expected exception for SwiGlu shape mismatch");
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

// ==========================================================================
// ElementwiseMul
// ==========================================================================
static bool testElementwiseMul() {
    mini_llama::Tensor a({3}, 0.0f);
    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    mini_llama::Tensor b({3}, 0.0f);
    b[0] = 4.0f;
    b[1] = 5.0f;
    b[2] = 6.0f;
    mini_llama::Tensor y = mini_llama::elementwiseMul(a, b);
    MINI_LLAMA_ASSERT_NEAR(y[0], 4.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(y[1], 10.0f, 1e-5f);
    MINI_LLAMA_ASSERT_NEAR(y[2], 18.0f, 1e-5f);
    return true;
}

static bool testElementwiseMulShapeMismatch() {
    mini_llama::Tensor a({3}, 0.0f);
    mini_llama::Tensor b({4}, 0.0f);

    mini_llama::elementwiseMul(a, b);
    MINI_LLAMA_ASSERT_FAIL(
        "expected exception for ElementwiseMul shape mismatch");

    return false;
}

// ==========================================================================
// argMax
// ==========================================================================
static bool testArgMax() {
    mini_llama::Tensor x({5}, 0.0f);
    x[0] = 1.0f;
    x[1] = 5.0f;
    x[2] = 3.0f;
    x[3] = 5.0f;
    x[4] = 2.0f;
    int idx = argMax(x);
    MINI_LLAMA_ASSERT_EQ(idx, 1);
    return true;
}

static bool testArgMaxEmpty() {
    mini_llama::Tensor x;
    argMax(x);
    MINI_LLAMA_ASSERT_FAIL(
        "expected exception for argMax on empty mini_llama::Tensor");

    return true;
}

// ==========================================================================
// TODO: test Rope
// ==========================================================================

static struct OpsTestRegistrar {
    OpsTestRegistrar() {
        registerTest("matmul_identity", testMatmulIdentity);
        registerTest("matmul_simple", testMatmulSimple);
        registerTest("matmul_rectangular", testMatmulRectangular);
        registerTest("matmul_shape_mismatch", testMatmulShapeMismatch);
        registerTest("linear_1d", testLinear1d);
        registerTest("linear_2d", testLinear2d);
        registerTest("linear_shape_mismatch", testLinearShapeMismatch);
        registerTest("rmsnorm_unit", testRmsNormUnit);
        registerTest("rmsnorm_scaled", testRmsNormScaled);
        registerTest("rmsnorm_with_weight", testRmsNormWithWeight);
        registerTest("rmsnorm_shape_mismatch", testRmsNormShapeMismatch);
        registerTest("Softmax", testSoftmax);
        registerTest("softmax_uniform", testSoftmaxUniform);
        registerTest("softmax_wrong_dim", testSoftmaxWrongDim);
        registerTest("softmax_empty", testSoftmaxEmpty);
        registerTest("Silu", testSilu);
        registerTest("SwiGlu", testSwiglu);
        registerTest("swiglu_shape_mismatch", testSwigluShapeMismatch);
        registerTest("ElementwiseMul", testElementwiseMul);
        registerTest("elementwise_mul_shape_mismatch",
                     testElementwiseMulShapeMismatch);
        registerTest("argMax", testArgMax);
        registerTest("argmax_empty", testArgMaxEmpty);
    }
} ops_test_registrar;