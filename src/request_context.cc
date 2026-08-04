// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/request_context.h"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace mini_llama {
namespace {

    std::atomic<unsigned long long> g_request_counter{0};

    std::string formatMs(double value) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(3) << value;
        return out.str();
    }

}  // namespace

// 记录当前时间
void RequestContext::start() { start_time = RequestClock::now(); }

void RequestContext::finish() { total_ms = elapsedMs(start_time); }

void RequestContext::recordEvent(const std::string& stage, double elapsed_ms,
                                 int tokens, const std::string& detail) {
    events.push_back({stage, elapsed_ms, tokens, detail});
}

void RequestContext::setError(const std::string& message) { error = message; }

bool RequestContext::ok() const { return error.empty(); }

std::string newTraceId() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    const unsigned long long seq = ++g_request_counter;

    std::tm tm_snapshot;
#if defined(_WIN32)
    localtime_s(&tm_snapshot, &time);
#else
    localtime_r(&time, &tm_snapshot);
#endif

    std::ostringstream out;
    out << "req_" << std::put_time(&tm_snapshot, "%Y%m%d%H%M%S") << "_"
        << std::setw(6) << std::setfill('0') << seq;
    return out.str();
}

double elapsedMs(RequestClock::time_point start) {
    return std::chrono::duration<double, std::milli>(RequestClock::now() -
                                                     start)
        .count();
}

// 开始一个请求上下文
RequestContext startRequest(const std::string& mode, const std::string& backend,
                            const std::string& model_path) {
    RequestContext request;
    request.trace_id = newTraceId();
    request.mode = mode;
    request.backend = backend;
    request.model_path = model_path;
    request.start();
    return request;
}

// 将summary格式化输出
std::string formatRequestTraceSummary(const RequestContext& request) {
    std::ostringstream out;
    out << "trace summary: trace_id=" << request.trace_id
        << " mode=" << request.mode
        << " status=" << (request.ok() ? "ok" : "error")
        << " backend=" << request.backend << " model=" << request.model_path
        << " prompt_tokens=" << request.prompt_tokens
        << " prefill_tokens=" << request.prefill_tokens
        << " decode_tokens=" << request.decode_tokens
        << " generated_tokens=" << request.generated_tokens
        << " model_load_ms=" << formatMs(request.model_load_ms)
        << " tokenize_ms=" << formatMs(request.tokenize_ms)
        << " prefill_ms=" << formatMs(request.prefill_ms)
        << " decode_ms=" << formatMs(request.decode_ms)
        << " sample_ms=" << formatMs(request.sample_ms)
        << " total_ms=" << formatMs(request.total_ms);
    if (!request.error.empty()) {
        out << " error=\"" << request.error << "\"";
    }
    return out.str();
}

// 将缓存的trace_event格式化输出
std::vector<std::string> formatRequestTraceEvents(
    const RequestContext& request) {
    std::vector<std::string> lines;
    lines.reserve(request.events.size());
    for (const TraceEvent& event : request.events) {
        std::ostringstream out;
        out << "trace event: trace_id=" << request.trace_id
            << " stage=" << event.stage << " tokens=" << event.tokens
            << " elapsed_ms=" << formatMs(event.elapsed_ms);
        if (!event.detail.empty()) {
            out << " detail=\"" << event.detail << "\"";
        }
        lines.push_back(out.str());
    }
    return lines;
}

}  // namespace mini_llama
