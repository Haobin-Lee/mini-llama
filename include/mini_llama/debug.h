// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_DEBUG_H_
#define INCLUDE_MINI_LLAMA_DEBUG_H_

#include <string>
#include <vector>

#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

struct BenchmarkResult {
    int n_prompt_tokens = 0;
    int n_generated_tokens = 0;
    int n_decode_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;

    double tokensPerSec() const {
        double total_ms = prefill_ms + decode_ms;
        return total_ms <= 0.0 ? 0.0 : n_generated_tokens * 1000.0 / total_ms;
    }
    double decodeTokensPerSec() const {
        if (decode_ms <= 0.0 || n_decode_tokens <= 0) {
            return 0.0;
        }
        return n_decode_tokens * 1000.0 / decode_ms;
    }
};

void dumpTensorShape(const Tensor& t, const std::string& name);
void dumpLogitsTopK(const Tensor& logits, int k);
void dumpKvCacheInfo(const KvCache& cache, int current_pos = -1);

// Run prefill + decode and return timing stats.
BenchmarkResult runBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens,
                             int n_predict, unsigned int seed, bool verbose);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_DEBUG_H_
