// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// debug 模块测试：BenchmarkResult 速率计算、runBenchmark 计数、dump 助手冒烟。

#include <string>
#include <vector>

#include "mini_llama/debug.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"
#include "mini_llama/tokenizer.h"
#include "tests/test_main.h"

using mini_llama::AsciiTokenizer;
using mini_llama::BenchmarkResult;
using mini_llama::KvCache;
using mini_llama::loadModel;
using mini_llama::MiniLlamaModel;
using mini_llama::Tensor;

static MiniLlamaModel loadTinyModelOrFail() {
    return loadModel("models/tiny/model.json", "models/tiny/model.bin");
}

// ---------------------------------------------------------------------------
// BenchmarkResult 数学
// ---------------------------------------------------------------------------
static bool testBenchmarkResultTokensPerSec() {
    BenchmarkResult r;
    r.n_generated_tokens = 100;
    r.prefill_ms = 100.0;
    r.decode_ms = 900.0;
    // 100 token / 1000 ms
    MINI_LLAMA_ASSERT_NEAR(r.tokensPerSec(), 100.0, 1e-9);

    // prefill + decode 为 0 时防止除零
    r.prefill_ms = 0.0;
    r.decode_ms = 0.0;
    MINI_LLAMA_ASSERT_NEAR(r.tokensPerSec(), 0.0, 1e-9);
    return true;
}

static bool testBenchmarkResultDecodeTokensPerSec() {
    BenchmarkResult r;
    r.n_decode_tokens = 50;
    r.decode_ms = 500.0;
    MINI_LLAMA_ASSERT_NEAR(r.decodeTokensPerSec(), 100.0, 1e-9);

    // decode_ms 为 0 时返回 0
    r.decode_ms = 0.0;
    MINI_LLAMA_ASSERT_NEAR(r.decodeTokensPerSec(), 0.0, 1e-9);

    // decode token 数为 0 时返回 0
    r.n_decode_tokens = 0;
    r.decode_ms = 500.0;
    MINI_LLAMA_ASSERT_NEAR(r.decodeTokensPerSec(), 0.0, 1e-9);
    return true;
}

// ---------------------------------------------------------------------------
// runBenchmark
// ---------------------------------------------------------------------------
static bool testRunBenchmarkCountsTokens() {
    MiniLlamaModel model = loadTinyModelOrFail();
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load tiny model");
    }

    AsciiTokenizer tokenizer;
    std::vector<int> prompt = tokenizer.encode("hello");

    BenchmarkResult r = mini_llama::runBenchmark(model, prompt, 3, 1u, false);

    MINI_LLAMA_ASSERT_EQ(r.n_prompt_tokens, static_cast<int>(prompt.size()));
    MINI_LLAMA_ASSERT_EQ(r.n_generated_tokens, 3);
    MINI_LLAMA_ASSERT_EQ(r.n_decode_tokens, 2);
    MINI_LLAMA_ASSERT_TRUE(r.prefill_ms >= 0.0);
    MINI_LLAMA_ASSERT_TRUE(r.decode_ms >= 0.0);
    MINI_LLAMA_ASSERT_TRUE(r.decodeTokensPerSec() > 0.0);
    return true;
}

static bool testRunBenchmarkSingleTokenBoundary() {
    MiniLlamaModel model = loadTinyModelOrFail();
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load tiny model");
    }

    AsciiTokenizer tokenizer;
    std::vector<int> prompt = tokenizer.encode("hi");

    // n_predict == 1：只生成 1 个 token，不进入 decode 循环
    BenchmarkResult r = mini_llama::runBenchmark(model, prompt, 1, 1u, false);
    MINI_LLAMA_ASSERT_EQ(r.n_generated_tokens, 1);
    MINI_LLAMA_ASSERT_EQ(r.n_decode_tokens, 0);
    MINI_LLAMA_ASSERT_NEAR(r.decodeTokensPerSec(), 0.0, 1e-9);

    // n_predict == 0：只做 prefill
    BenchmarkResult r0 = mini_llama::runBenchmark(model, prompt, 0, 1u, false);
    MINI_LLAMA_ASSERT_EQ(r0.n_generated_tokens, 0);
    MINI_LLAMA_ASSERT_EQ(r0.n_decode_tokens, 0);
    MINI_LLAMA_ASSERT_TRUE(r0.prefill_ms >= 0.0);
    return true;
}

// ---------------------------------------------------------------------------
// dump 助手冒烟（仅验证不崩溃）
// ---------------------------------------------------------------------------
static bool testDebugDumpHelpersSmoke() {
    Tensor t({2, 3}, 0.5f);
    mini_llama::dumpTensorShape(t, "smoke");

    Tensor logits({5}, 0.0f);
    for (int i = 0; i < 5; ++i) {
        logits[i] = static_cast<float>(i) * 0.1f;
    }
    mini_llama::dumpLogitsTopK(logits, 3);
    mini_llama::dumpLogitsTopK(logits, 0);  // k<=0 时全量
    mini_llama::dumpLogitsTopK(t, 3);       // 非 1D 走无效形状分支

    KvCache cache(1, 4, 2, 8);
    mini_llama::dumpKvCacheInfo(cache, 2);
    mini_llama::dumpKvCacheInfo(cache);
    return true;
}

// ---------------------------------------------------------------------------
// 注册
// ---------------------------------------------------------------------------
static struct DebugTestRegistrar {
    DebugTestRegistrar() {
        registerTest("benchmark_result_tokens_per_sec",
                     testBenchmarkResultTokensPerSec);
        registerTest("benchmark_result_decode_tokens_per_sec",
                     testBenchmarkResultDecodeTokensPerSec);
        registerTest("run_benchmark_counts_tokens",
                     testRunBenchmarkCountsTokens);
        registerTest("run_benchmark_single_token_boundary",
                     testRunBenchmarkSingleTokenBoundary);
        registerTest("debug_dump_helpers_smoke", testDebugDumpHelpersSmoke);
    }
} debug_test_registrar;
