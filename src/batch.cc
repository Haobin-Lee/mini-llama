// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/batch.h"

#include "errlog/bizlog.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/model.h"

namespace mini_llama {

MiniBatch MiniBatch::single(int token, int pos) {
    MiniBatch batch;
    batch.tokens.push_back(token);
    batch.positions.push_back(pos);
    return batch;
}

MiniBatch MiniBatch::fromTokens(const std::vector<int>& toks, int start_pos) {
    MiniBatch batch;
    batch.tokens = toks;
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.positions.push_back(start_pos + static_cast<int>(i));
    }
    return batch;
}

// Processes all tokens in the batch sequentially.
// Returns logits for the *last* token in the batch.
// This unifies prefill (multi-token) and Decode (single-token) paths.
Tensor forwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch) {
    Tensor logits;
    if (batch.tokens.empty()) {
        BIZLOG(ErrorCode::kInvalidBatchSize, "Batch is empty");
        return logits;
    }

    if (batch.tokens.size() != batch.positions.size()) {
        BIZLOG(ErrorCode::kInvalidBatchSize,
               "Batch token and position sizes must match");
        return logits;
    }

    for (size_t i = 0; i < batch.tokens.size(); ++i) {
        ctx.pos = batch.positions[i];
        logits = forwardToken(ctx, model, batch.tokens[i]);
        if (logits.empty()) {
            return logits;
        }
        ctx.token_history.push_back(batch.tokens[i]);
    }
    return logits;
}

}  // namespace mini_llama
