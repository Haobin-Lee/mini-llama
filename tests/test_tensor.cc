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

// TODO: add tests for flatIndex, at, assertShape, reshapeChecked.
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
        registerTest("tensor_at1", testTensorAt1);
        registerTest("tensor_at1_wrong_dim", testTensorAt1WrongDim);
        registerTest("tensor_at2", testTensorAt2);
        registerTest("tensor_at3", testTensorAt3);
        registerTest("tensor_at4", testTensorAt4);
    }
} tensor_test_registrar;