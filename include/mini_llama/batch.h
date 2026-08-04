// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_BATCH_H_
#define INCLUDE_MINI_LLAMA_BATCH_H_

#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/model.h"

namespace mini_llama {

// A collection of tokens with positions (mirrors llama_batch).
struct MiniBatch {
    std::vector<int> tokens;
    std::vector<int> positions;

    int numTokens() const { return static_cast<int>(tokens.size()); }

    // Single-token batch (decode step).
    static MiniBatch single(int token, int pos);

    // Multi-token batch (prefill step); positions start at start_pos.
    static MiniBatch fromTokens(const std::vector<int>& toks,
                                int start_pos = 0);
};

// Forward pass over a batch. Processes tokens sequentially and returns the
// logits for the last token (unifies prefill and decode).
Tensor forwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_BATCH_H_
