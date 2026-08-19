// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_GGUF_LOADER_H_
#define INCLUDE_MINI_LLAMA_GGUF_LOADER_H_

#include <string>

#include "mini_llama/model.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// Load a model from a GGUF file.
// Supports F32, Q8_0, Q4_0, and Q4_1 tensors. Norm, embedding, and bias
// tensors are converted to F32 at load time when needed.
// ---------------------------------------------------------------------------
MiniLlamaModel loadGgufModel(const std::string& gguf_path);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_GGUF_LOADER_H_
