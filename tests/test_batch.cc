// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "tests/test_main.h"

using mini_llama::MiniBatch;
using mini_llama::MiniLlamaModel;
using namespace mini_llama;
// ---------------------------------------------------------------------------
// MiniBatch construction
// ---------------------------------------------------------------------------
static bool testBatchSingle() {
    MiniBatch b = MiniBatch::single(42, 5);
    MINI_LLAMA_ASSERT_EQ(b.numTokens(), 1);
    MINI_LLAMA_ASSERT_EQ(b.tokens[0], 42);
    MINI_LLAMA_ASSERT_EQ(b.positions[0], 5);
    return true;
}

static bool testBatchFromTokens() {
    std::vector<int> toks = {10, 20, 30};
    MiniBatch b = MiniBatch::fromTokens(toks, 0);
    MINI_LLAMA_ASSERT_EQ(b.numTokens(), 3);
    MINI_LLAMA_ASSERT_EQ(b.tokens[0], 10);
    MINI_LLAMA_ASSERT_EQ(b.tokens[1], 20);
    MINI_LLAMA_ASSERT_EQ(b.tokens[2], 30);
    MINI_LLAMA_ASSERT_EQ(b.positions[0], 0);
    MINI_LLAMA_ASSERT_EQ(b.positions[1], 1);
    MINI_LLAMA_ASSERT_EQ(b.positions[2], 2);
    return true;
}

static bool testBatchFromTokensWithOffset() {
    std::vector<int> toks = {5, 6};
    MiniBatch b = MiniBatch::fromTokens(toks, 10);
    MINI_LLAMA_ASSERT_EQ(b.positions[0], 10);
    MINI_LLAMA_ASSERT_EQ(b.positions[1], 11);
    return true;
}

static bool TestBatchEmpty() {
    MiniBatch b;
    MINI_LLAMA_ASSERT_EQ(b.numTokens(), 0);
    return true;
}

// ---------------------------------------------------------------------------
// ForwardBatch
// ---------------------------------------------------------------------------
static bool testForwardBatchPrefillMatchesIndividual() {
    MiniLlamaModel model =
        loadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    std::vector<int> tokens = {1, 104, 101, 108, 108, 111};

    // Individual forward
    MiniLlamaContext ctx1(&model);
    Tensor logits1;
    for (size_t i = 0; i < tokens.size(); ++i) {
        ctx1.pos = static_cast<int>(i);
        logits1 = forwardToken(ctx1, model, tokens[i]);
        ctx1.token_history.push_back(tokens[i]);
    }

    // Batch forward
    MiniLlamaContext ctx2(&model);
    MiniBatch batch = MiniBatch::fromTokens(tokens, 0);
    Tensor logits2 = forwardBatch(ctx2, model, batch);

    MINI_LLAMA_ASSERT_EQ(logits1.numDims(), 1);
    MINI_LLAMA_ASSERT_EQ(logits2.numDims(), 1);
    MINI_LLAMA_ASSERT_EQ(logits1.shape[0], logits2.shape[0]);
    for (size_t i = 0; i < logits1.size(); ++i) {
        MINI_LLAMA_ASSERT_NEAR(logits1.data[i], logits2.data[i], 1e-6f);
    }
    return true;
}

static bool testForwardBatchSingleMatchesIndividual() {
    MiniLlamaModel model =
        loadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    // Individual forward
    MiniLlamaContext ctx1(&model);
    ctx1.pos = 3;
    Tensor logits1 = forwardToken(ctx1, model, 42);

    // Batch forward with single token
    MiniLlamaContext ctx2(&model);
    MiniBatch batch = MiniBatch::single(42, 3);
    Tensor logits2 = forwardBatch(ctx2, model, batch);

    MINI_LLAMA_ASSERT_EQ(logits1.numDims(), 1);
    MINI_LLAMA_ASSERT_EQ(logits2.numDims(), 1);
    MINI_LLAMA_ASSERT_EQ(logits1.shape[0], logits2.shape[0]);
    for (size_t i = 0; i < logits1.size(); ++i) {
        MINI_LLAMA_ASSERT_NEAR(logits1.data[i], logits2.data[i], 1e-6f);
    }
    return true;
}

static bool testForwardBatchUpdatesContext() {
    MiniLlamaModel model =
        loadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    std::vector<int> tokens = {1, 104, 101};
    MiniLlamaContext ctx(&model);
    MiniBatch batch = MiniBatch::fromTokens(tokens, 0);
    forwardBatch(ctx, model, batch);

    MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        MINI_LLAMA_ASSERT_EQ(ctx.token_history[i], tokens[i]);
    }
    MINI_LLAMA_ASSERT_EQ(ctx.pos, static_cast<int>(tokens.size()) - 1);
    return true;
}

static bool testForwardBatchAppendsDecodeHistory() {
    MiniLlamaModel model =
        loadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    MiniLlamaContext ctx(&model);
    MiniBatch prefill = MiniBatch::fromTokens({1, 104}, 0);
    forwardBatch(ctx, model, prefill);

    MiniBatch decode = MiniBatch::single(101, 2);
    forwardBatch(ctx, model, decode);

    MINI_LLAMA_ASSERT_EQ(ctx.token_history.size(), 3);
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[0], 1);
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[1], 104);
    MINI_LLAMA_ASSERT_EQ(ctx.token_history[2], 101);
    MINI_LLAMA_ASSERT_EQ(ctx.pos, 2);
    return true;
}

static bool testForwardBatchEmptyRejected() {
    MiniLlamaModel model =
        loadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    MiniLlamaContext ctx(&model);
    MiniBatch batch;
    try {
        forwardBatch(ctx, model, batch);
        MINI_LLAMA_ASSERT_FAIL("expected exception for empty batch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

static bool testForwardBatchMismatchedSizesRejected() {
    MiniLlamaModel model =
        loadModel("models/tiny/model.json", "models/tiny/model.bin");
    if (!model.loaded) {
        MINI_LLAMA_ASSERT_FAIL("failed to load model");
    }

    MiniLlamaContext ctx(&model);
    MiniBatch batch;
    batch.tokens = {1, 2};
    batch.positions = {0};
    try {
        forwardBatch(ctx, model, batch);
        MINI_LLAMA_ASSERT_FAIL("expected exception for size mismatch");
    } catch (const std::runtime_error&) {
        // expected
    }
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct BatchTestRegistrar {
    BatchTestRegistrar() {
        registerTest("batch_single", testBatchSingle);
        registerTest("batch_from_tokens", testBatchFromTokens);
        registerTest("batch_from_tokens_with_offset",
                     testBatchFromTokensWithOffset);
        registerTest("batch_empty", TestBatchEmpty);
        registerTest("forward_batch_prefill_matches_individual",
                     testForwardBatchPrefillMatchesIndividual);
        registerTest("forward_batch_single_matches_individual",
                     testForwardBatchSingleMatchesIndividual);
        registerTest("forward_batch_updates_context",
                     testForwardBatchUpdatesContext);
        registerTest("forward_batch_appends_decode_history",
                     testForwardBatchAppendsDecodeHistory);
        registerTest("forward_batch_empty_rejected",
                     testForwardBatchEmptyRejected);
        registerTest("forward_batch_mismatched_sizes_rejected",
                     testForwardBatchMismatchedSizesRejected);
    }
} batch_test_registrar;
