// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_SAMPLER_H_
#define INCLUDE_MINI_LLAMA_SAMPLER_H_

#include <random>

#include "mini_llama/tensor.h"

namespace mini_llama {

struct SamplingParams {
    float temperature = 0.0f;
    int top_k = 0;          // 0 = disabled
    unsigned int seed = 0;  // 0 = random device
};

// Greedy, temperature, and top-k sampling.
class MiniSampler {
public:
    explicit MiniSampler(unsigned int seed = 0);
    explicit MiniSampler(const SamplingParams& params);

    int sample(const Tensor& logits, const SamplingParams& params);

    static int sampleGreedy(const Tensor& logits);
    int sampleTemperature(const Tensor& logits, float temperature);
    int sampleTopK(const Tensor& logits, float temperature, int top_k);

private:
    std::mt19937 rng_;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_SAMPLER_H_
