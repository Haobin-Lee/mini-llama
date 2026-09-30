// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// Tests for Tensor. One example is provided; add more (indexing, bounds
// checks, reshape, row pointers, 3D/4D accessors ...).

#include "mini_llama/tensor.h"
#include "tests/test_main.h"

using mini_llama::makeTensor2D;
using mini_llama::Tensor;

static bool testTensorShapeAndSize() {
    Tensor t1({3, 4, 5}, 0.0f);
    MINI_LLAMA_ASSERT_EQ(t1.numDims(), 3);
    MINI_LLAMA_ASSERT_EQ(t1.size(), 60);
    MINI_LLAMA_ASSERT_EQ(t1.numElements(), 60);
    MINI_LLAMA_ASSERT_EQ(t1.shape[0], 3);
    MINI_LLAMA_ASSERT_EQ(t1.shape[1], 4);
    MINI_LLAMA_ASSERT_EQ(t1.shape[2], 5);
    return true;
}

static bool testTensorIndexing() {
    Tensor t({2, 3}, 0.0f);
    t.at({0, 0}) = 1.0f;
    t.at({0, 1}) = 2.0f;
    t.at({1, 2}) = 6.0f;
    MINI_LLAMA_ASSERT_NEAR(t.at({0, 0}), 1.0f, 1e-6f);
    MINI_LLAMA_ASSERT_NEAR(t.at({0, 1}), 2.0f, 1e-6f);
    MINI_LLAMA_ASSERT_NEAR(t.at({1, 2}), 6.0f, 1e-6f);
    return true;
}

static bool testTensorFill() {
    Tensor t({10}, 3.14f);
    MINI_LLAMA_ASSERT_EQ(t.size(), 10);
    for (size_t i = 0; i < t.size(); ++i) {
        MINI_LLAMA_ASSERT_NEAR(t.data[i], 3.14f, 1e-6f);
    }
    return true;
}

static bool testTensorIndexTooLarge() {
    Tensor t({2, 3}, 0.0f);
    t.at({0, 3});

    return false;
}

// flatIndex 回归测试：泛型 at() 必须与手写 at1/at2/at3/at4 结果一致。
// 历史 bug：循环条件 axis > 0 导致第一维（axis=0）索引从不参与累加。
static bool testTensorFlatIndexMatchesAt2() {
    Tensor t({2, 3}, 0.0f);
    t.at({1, 2}) = 6.0f;
    // 手写公式展开：flat = 1 * 3 + 2 = 5
    MINI_LLAMA_ASSERT_NEAR(t.at({1, 2}), 6.0f, 1e-6f);
    MINI_LLAMA_ASSERT_NEAR(t.data[5], 6.0f, 1e-6f);
    MINI_LLAMA_ASSERT_NEAR(t.at2(1, 2), t.at({1, 2}), 1e-6f);
    MINI_LLAMA_ASSERT_EQ(t.flatIndex({1, 2}), static_cast<size_t>(5));

    // 反向：用 at2 写入，用 at 读取
    t.at2(0, 1) = 7.0f;
    MINI_LLAMA_ASSERT_NEAR(t.at({0, 1}), 7.0f, 1e-6f);
    return true;
}

static bool testTensorFlatIndexMatchesAt3() {
    Tensor t({2, 3, 4}, 0.0f);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                float v = static_cast<float>(i * 100 + j * 10 + k);
                t.at3(i, j, k) = v;
                MINI_LLAMA_ASSERT_NEAR(t.at({i, j, k}), v, 1e-6f);
            }
        }
    }
    return true;
}

static bool testTensorFlatIndexMatchesAt4() {
    Tensor t({2, 2, 2, 3}, 0.0f);
    t.at4(1, 1, 1, 2) = 9.0f;
    MINI_LLAMA_ASSERT_NEAR(t.at({1, 1, 1, 2}), 9.0f, 1e-6f);
    MINI_LLAMA_ASSERT_NEAR(t.data[((1 * 2 + 1) * 2 + 1) * 3 + 2], 9.0f, 1e-6f);
    return true;
}

static bool testTensorFlatIndex1d() {
    Tensor t({5}, 0.0f);
    t.at({3}) = 4.0f;
    MINI_LLAMA_ASSERT_NEAR(t.at1(3), 4.0f, 1e-6f);
    MINI_LLAMA_ASSERT_EQ(t.flatIndex({3}), static_cast<size_t>(3));
    return true;
}

static bool testTensorIsSameShape() {
    Tensor t({2, 3}, 0.0f);
    MINI_LLAMA_ASSERT_TRUE(t.isSameShape({2, 3}, "test"));
    MINI_LLAMA_ASSERT_TRUE(!t.isSameShape({3, 2}, "test"));
    MINI_LLAMA_ASSERT_TRUE(!t.isSameShape({2, 3, 1}, "test"));
    return true;
}

static bool testTensorReshapeChecked() {
    Tensor t({2, 3}, 0.0f);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data[i] = static_cast<float>(i);
    }

    Tensor r = t.reshapeChecked({3, 2}, "test");
    MINI_LLAMA_ASSERT_TRUE(r.isSameShape({3, 2}, "test"));
    MINI_LLAMA_ASSERT_EQ(r.size(), t.size());
    for (size_t i = 0; i < r.size(); ++i) {
        MINI_LLAMA_ASSERT_NEAR(r.data[i], static_cast<float>(i), 1e-6f);
    }

    // 元素数量不一致时必须返回空张量
    Tensor bad = t.reshapeChecked({4, 2}, "test");
    MINI_LLAMA_ASSERT_TRUE(bad.empty());
    return true;
}

static bool testTensorAt1() {
    Tensor t({5}, 0.0f);
    t.at1(2) = 7.0f;
    MINI_LLAMA_ASSERT_NEAR(t.at1(2), 7.0f, 1e-6f);
    return true;
}

static bool testTensorAt1WrongDim() {
    Tensor t({2, 3}, 0.0f);
    t.at1(0);
    return false;
}

static bool testTensorAt2() {
    Tensor t({2, 3}, 0.0f);
    t.at2(0, 1) = 5.0f;
    t.at2(1, 2) = 9.0f;
    MINI_LLAMA_ASSERT_NEAR(t.at2(0, 1), 5.0f, 1e-6f);
    MINI_LLAMA_ASSERT_NEAR(t.at2(1, 2), 9.0f, 1e-6f);
    return true;
}

static bool testTensorAt3() {
    Tensor t({2, 2, 2}, 0.0f);
    t.at3(1, 0, 1) = 3.0f;
    MINI_LLAMA_ASSERT_NEAR(t.at3(1, 0, 1), 3.0f, 1e-6f);
    return true;
}

static bool testTensorAt4() {
    Tensor t({2, 2, 2, 2}, 0.0f);
    t.at4(1, 1, 0, 0) = 4.0f;
    MINI_LLAMA_ASSERT_NEAR(t.at4(1, 1, 0, 0), 4.0f, 1e-6f);
    return true;
}

// 在test main函数前注册所有测试用例
static struct TensorTestRegistrar {
    TensorTestRegistrar() {
        registerTest("tensor_shape_and_size", testTensorShapeAndSize);
        registerTest("tensor_indexing", testTensorIndexing);
        registerTest("tensor_fill", testTensorFill);
        registerTest("tensor_index_too_large", testTensorIndexTooLarge);
        registerTest("tensor_flat_index_matches_at2",
                     testTensorFlatIndexMatchesAt2);
        registerTest("tensor_flat_index_matches_at3",
                     testTensorFlatIndexMatchesAt3);
        registerTest("tensor_flat_index_matches_at4",
                     testTensorFlatIndexMatchesAt4);
        registerTest("tensor_flat_index_1d", testTensorFlatIndex1d);
        registerTest("tensor_is_same_shape", testTensorIsSameShape);
        registerTest("tensor_reshape_checked", testTensorReshapeChecked);
        registerTest("tensor_at1", testTensorAt1);
        registerTest("tensor_at1_wrong_dim", testTensorAt1WrongDim);
        registerTest("tensor_at2", testTensorAt2);
        registerTest("tensor_at3", testTensorAt3);
        registerTest("tensor_at4", testTensorAt4);
    }
} tensor_test_registrar;