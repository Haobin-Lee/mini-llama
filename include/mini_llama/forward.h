// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_FORWARD_H_
#define INCLUDE_MINI_LLAMA_FORWARD_H_

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"

namespace mini_llama {

// Forward pass for a single token. Returns logits: [vocab_size].
Tensor forwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_FORWARD_H_
