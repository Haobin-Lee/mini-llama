// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_
#define INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_

#include <chrono>
#include <string>
#include <vector>

namespace mini_llama {

using RequestClock = std::chrono::steady_clock;

struct TraceEvent {
    std::string stage;
    double elapsed_ms = 0.0;
    int tokens = 0;
    std::string detail;
};

// Request-level context for tracing one user request from entry to finish.
struct RequestContext {
    std::string trace_id;
    std::string mode;
    std::string backend;
    std::string model_path;

    RequestClock::time_point start_time;
    double total_ms = 0.0;

    // 记录每个阶段的token数
    int prompt_tokens = 0;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    int generated_tokens = 0;

    // 记录每个阶段的时间
    double model_load_ms = 0.0;
    double tokenize_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double sample_ms = 0.0;

    std::string error;
    std::vector<TraceEvent> events;

    void start();
    void finish();
    void recordEvent(const std::string& stage, double elapsed_ms,
                     int tokens = 0, const std::string& detail = "");
    void setError(const std::string& message);
    bool ok() const;
};

std::string newTraceId();
double elapsedMs(RequestClock::time_point start);  // 总耗时
RequestContext startRequest(const std::string& mode, const std::string& backend,
                            const std::string& model_path);
std::string formatRequestTraceSummary(
    const RequestContext& request);  // 汇总结果
std::vector<std::string> formatRequestTraceEvents(
    const RequestContext& request);  // 记录每次事件

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_
