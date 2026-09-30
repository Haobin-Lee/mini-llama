// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// MiniSampler 测试：greedy / temperature / top-k、种子可复现、非法参数拒绝。
// 注意：采样器对非法输入不抛异常，而是返回 -1（与 ops 的 BIZLOG 风格一致）。

#include <limits>
#include <vector>

#include "mini_llama/sampler.h"
#include "mini_llama/tensor.h"
#include "tests/test_main.h"

using mini_llama::MiniSampler;
using mini_llama::SamplingParams;
using mini_llama::Tensor;

// ---------------------------------------------------------------------------
// Greedy
// ---------------------------------------------------------------------------
static bool testSamplerGreedyMatchesArgMax() {
    Tensor logits({5}, 0.0f);
    logits[0] = 1.0f;
    logits[1] = 5.0f;
    logits[2] = 3.0f;
    logits[3] = 5.0f;
    logits[4] = 2.0f;

    MINI_LLAMA_ASSERT_EQ(MiniSampler::sampleGreedy(logits), 1);

    // 并列最大值取第一个下标（argMax 使用严格大于比较）
    Tensor tie({3}, 0.0f);
    tie[0] = 5.0f;
    tie[1] = 5.0f;
    tie[2] = 1.0f;
    MINI_LLAMA_ASSERT_EQ(MiniSampler::sampleGreedy(tie), 0);
    return true;
}

static bool testSamplerGreedyOn2dLogits() {
    // greedy 按扁平数据取 argmax（logits 通常为 1D，但 2D 输入不报错）
    Tensor logits({2, 3}, 0.0f);
    logits[3] = 7.0f;  // 扁平下标 3 == (1, 0)
    MINI_LLAMA_ASSERT_EQ(MiniSampler::sampleGreedy(logits), 3);
    return true;
}

// ---------------------------------------------------------------------------
// temperature / top-k 与 greedy 的等价关系
// ---------------------------------------------------------------------------
static bool testSamplerZeroTemperatureIsGreedy() {
    Tensor logits({3}, 0.0f);
    logits[0] = 1.0f;
    logits[1] = 4.0f;
    logits[2] = 2.0f;

    MiniSampler sampler(7);
    MINI_LLAMA_ASSERT_EQ(sampler.sampleTemperature(logits, 0.0f), 1);

    SamplingParams params;  // 默认 temperature = 0
    MINI_LLAMA_ASSERT_EQ(sampler.sample(logits, params), 1);
    return true;
}

static bool testSamplerTopKOneIsGreedy() {
    Tensor logits({3}, 0.0f);
    logits[0] = 1.0f;
    logits[1] = 4.0f;
    logits[2] = 2.0f;

    MiniSampler sampler(11);
    MINI_LLAMA_ASSERT_EQ(sampler.sampleTopK(logits, 1.0f, 1), 1);

    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 1;
    MINI_LLAMA_ASSERT_EQ(sampler.sample(logits, params), 1);
    return true;
}

// ---------------------------------------------------------------------------
// 随机采样行为
// ---------------------------------------------------------------------------
static bool testSamplerSeedReproducibility() {
    Tensor logits({4}, 0.0f);
    logits[0] = 0.1f;
    logits[1] = 0.2f;
    logits[2] = 0.3f;
    logits[3] = 0.4f;

    MiniSampler a(12345);
    MiniSampler b(12345);

    std::vector<int> seq_a;
    std::vector<int> seq_b;
    for (int i = 0; i < 32; ++i) {
        seq_a.push_back(a.sampleTemperature(logits, 1.0f));
        seq_b.push_back(b.sampleTemperature(logits, 1.0f));
    }

    MINI_LLAMA_ASSERT_EQ(seq_a.size(), seq_b.size());
    for (size_t i = 0; i < seq_a.size(); ++i) {
        MINI_LLAMA_ASSERT_EQ(seq_a[i], seq_b[i]);
    }
    return true;
}

static bool testSamplerTemperatureProducesVariedTokens() {
    Tensor logits({4}, 0.0f);  // 均匀分布
    MiniSampler sampler(99);

    std::vector<int> counts(4, 0);
    for (int i = 0; i < 64; ++i) {
        int token = sampler.sampleTemperature(logits, 1.0f);
        MINI_LLAMA_ASSERT_TRUE(token >= 0 && token < 4);
        counts[static_cast<size_t>(token)]++;
    }

    int distinct = 0;
    for (int c : counts) {
        if (c > 0) {
            ++distinct;
        }
    }
    // 64 次均匀采样全部落在同一个 token 的概率可忽略
    MINI_LLAMA_ASSERT_TRUE(distinct >= 2);
    return true;
}

static bool testSamplerTopKRestrictsCandidates() {
    Tensor logits({4}, 0.0f);
    logits[0] = 10.0f;
    logits[1] = 9.0f;
    logits[2] = -10.0f;
    logits[3] = -20.0f;

    MiniSampler sampler(5);
    for (int i = 0; i < 32; ++i) {
        int token = sampler.sampleTopK(logits, 1.0f, 2);
        MINI_LLAMA_ASSERT_TRUE(token == 0 || token == 1);
    }

    // top_k 大于词表时自动截断，不应越界
    for (int i = 0; i < 8; ++i) {
        int token = sampler.sampleTopK(logits, 1.0f, 100);
        MINI_LLAMA_ASSERT_TRUE(token >= 0 && token < 4);
    }
    return true;
}

// ---------------------------------------------------------------------------
// 非法输入（返回 -1，不抛异常）
// ---------------------------------------------------------------------------
static bool testSamplerRejectsInvalidInputs() {
    MiniSampler sampler(1);
    Tensor empty;

    MINI_LLAMA_ASSERT_EQ(MiniSampler::sampleGreedy(empty), -1);
    MINI_LLAMA_ASSERT_EQ(sampler.sampleTemperature(empty, 1.0f), -1);
    MINI_LLAMA_ASSERT_EQ(sampler.sampleTopK(empty, 1.0f, 2), -1);

    Tensor logits({3}, 0.0f);
    logits[0] = 1.0f;
    logits[1] = 2.0f;
    logits[2] = 3.0f;

    // 负 temperature / 负 top_k
    MINI_LLAMA_ASSERT_EQ(sampler.sampleTemperature(logits, -0.5f), -1);
    MINI_LLAMA_ASSERT_EQ(sampler.sampleTopK(logits, 1.0f, -1), -1);
    MINI_LLAMA_ASSERT_EQ(sampler.sampleTopK(logits, -1.0f, 2), -1);
    // sampleTopK 要求 top_k 严格为正（0 视为非法，而非"禁用"）
    MINI_LLAMA_ASSERT_EQ(sampler.sampleTopK(logits, 1.0f, 0), -1);

    SamplingParams params;
    params.temperature = -1.0f;
    MINI_LLAMA_ASSERT_EQ(sampler.sample(logits, params), -1);
    params = SamplingParams();
    params.top_k = -1;
    MINI_LLAMA_ASSERT_EQ(sampler.sample(logits, params), -1);

    // 非有限 logits（NaN）
    Tensor nan_logits({3}, 0.0f);
    nan_logits[1] = std::numeric_limits<float>::quiet_NaN();
    MINI_LLAMA_ASSERT_EQ(MiniSampler::sampleGreedy(nan_logits), -1);
    return true;
}

// ---------------------------------------------------------------------------
// 注册
// ---------------------------------------------------------------------------
static struct SamplerTestRegistrar {
    SamplerTestRegistrar() {
        registerTest("sampler_greedy_matches_argmax",
                     testSamplerGreedyMatchesArgMax);
        registerTest("sampler_greedy_on_2d_logits",
                     testSamplerGreedyOn2dLogits);
        registerTest("sampler_zero_temperature_is_greedy",
                     testSamplerZeroTemperatureIsGreedy);
        registerTest("sampler_top_k_one_is_greedy", testSamplerTopKOneIsGreedy);
        registerTest("sampler_seed_reproducibility",
                     testSamplerSeedReproducibility);
        registerTest("sampler_temperature_produces_varied_tokens",
                     testSamplerTemperatureProducesVariedTokens);
        registerTest("sampler_top_k_restricts_candidates",
                     testSamplerTopKRestrictsCandidates);
        registerTest("sampler_rejects_invalid_inputs",
                     testSamplerRejectsInvalidInputs);
    }
} sampler_test_registrar;
