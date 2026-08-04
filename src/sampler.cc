// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//

#include "mini_llama/sampler.h"

#include <stdexcept>

#include "errlog/bizlog.h"
#include "mini_llama/ops.h"
#include "mini_llama/tensor.h"

namespace mini_llama {
using errlog::ErrorCode;

namespace {

    constexpr float kGreedyTemperature = 1e-6f;

    bool validateLogits(const Tensor& logits) {
        if (logits.empty()) {
            BIZLOG(ErrorCode::kTensorShapeEmpty,
                   "sampler: cannot sample from empty logits");
            return false;
        }
        for (float value : logits.data) {
            if (!std::isfinite(value)) {
                BIZLOG(ErrorCode::kTensorValueOutofRange,
                       "sampler: logits must be finite");
                return false;
            }
        }

        return true;
    }

    bool validateTemperature(float temperature) {
        if (!std::isfinite(temperature) || temperature < 0.0f) {
            BIZLOG(ErrorCode::kTensorValueOutofRange,
                   "sampler: temperature must be finite and non-negative");
            return false;
        }
        return true;
    }

    bool validateTopK(int top_k) {
        if (top_k < 0) {
            BIZLOG(ErrorCode::kTensorValueOutofRange,
                   "sampler: top_k must be non-negative");
            return false;
        }
        return true;
    }

    int sampleFromCandidates(
        const std::vector<std::pair<float, int>>& candidates, float temperature,
        std::mt19937& rng) {
        if (temperature < kGreedyTemperature) {
            int best = candidates[0].second;
            float best_val = candidates[0].first;
            for (size_t i = 1; i < candidates.size(); ++i) {
                if (candidates[i].first > best_val) {
                    best = candidates[i].second;
                    best_val = candidates[i].first;
                }
            }
            return best;
        }

        float max_val = candidates[0].first;
        for (size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].first > max_val) {
                max_val = candidates[i].first;
            }
        }

        std::vector<float> probs(candidates.size(), 0.0f);
        float sum = 0.0f;
        for (size_t i = 0; i < candidates.size(); ++i) {
            probs[i] = std::exp((candidates[i].first - max_val) / temperature);
            sum += probs[i];
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float cumulative = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumulative += probs[i] / sum;
            if (r <= cumulative) {
                return candidates[i].second;
            }
        }
        return candidates.back().second;
    }
}  // namespace

// NOTE: rng_ must be initialized in the member init list once you implement
MiniSampler::MiniSampler(unsigned int seed) : rng_(seed) {
    if (seed != 0) {
        rng_.seed(seed);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}
MiniSampler::MiniSampler(const SamplingParams& params)
    : MiniSampler(params.seed) {}

int MiniSampler::sampleGreedy(const Tensor& logits) {
    if (!validateLogits(logits)) {
        return -1;
    }
    return argMax(logits);
}
int MiniSampler::sampleTemperature(const Tensor& logits, float temperature) {
    if (!validateLogits(logits)) {
        return -1;
    }
    if (!validateTemperature(temperature)) {
        return -1;
    }
    if (temperature < kGreedyTemperature) {
        return sampleGreedy(logits);
    }

    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(logits.data.size());
    for (size_t i = 0; i < logits.data.size(); ++i) {
        candidates.emplace_back(logits.data[i], static_cast<int>(i));
    }
    return sampleFromCandidates(candidates, temperature, rng_);
}
int MiniSampler::sampleTopK(const Tensor& logits, float temperature,
                            int top_k) {
    if (!validateLogits(logits)) {
        return -1;
    }
    if (!validateTemperature(temperature)) {
        return -1;
    }
    if (!validateTopK(top_k)) {
        return -1;
    }
    if (top_k == 0) {
        BIZLOG(ErrorCode::kInvalidParam,
               "sampler: top_k must be positive when calling SampleTopK");
        return -1;
    }
    if (top_k == 1) {
        return sampleGreedy(logits);
    }
    if (top_k > static_cast<int>(logits.data.size())) {
        top_k = static_cast<int>(logits.data.size());
    }

    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(logits.data.size());
    for (size_t i = 0; i < logits.data.size(); ++i) {
        candidates.emplace_back(logits.data[i], static_cast<int>(i));
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  if (a.first == b.first) {
                      return a.second < b.second;
                  }
                  return a.first > b.first;
              });
    candidates.resize(static_cast<size_t>(top_k));
    return sampleFromCandidates(candidates, temperature, rng_);
}

int MiniSampler::sample(const Tensor& logits, const SamplingParams& params) {
    if (!validateTemperature(params.temperature)) {
        return -1;
    }
    if (!validateTopK(params.top_k)) {
        return -1;
    }
    if (params.temperature < kGreedyTemperature || params.top_k == 1) {
        return sampleGreedy(logits);
    }
    if (params.top_k > 1) {
        return sampleTopK(logits, params.temperature, params.top_k);
    }
    return sampleTemperature(logits, params.temperature);
}

int SampleGreedy(const Tensor& logits) {
    return MiniSampler::sampleGreedy(logits);
}

}  // namespace mini_llama
