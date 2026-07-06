// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "errlog/log_init.h"

#include <spdlog/async.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <mutex>
#include <vector>

namespace errlog {

namespace {
std::once_flag g_init_once;  // 保证只初始化一次
bool g_async = false;        // 记录是否为异步模式，退出时按需处理
}  // namespace

void initLogger(const LogConfig& config) {
    std::call_once(g_init_once, [&config]() {
        // ---- 1) 组装 sink（统一使用 *_mt 多线程安全版本）----
        std::vector<spdlog::sink_ptr> sinks;

        // 控制台彩色 sink
        if (config.enable_console) {
            auto console_sink =
                std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            sinks.push_back(console_sink);
        }

        // 按天切割文件 sink：每天 rotate_hour:rotate_minute 生成新文件，
        // 最多保留 keep_days 个（第三个 true 表示 truncate=已存在则截断）。
        // daily_file_sink 会在写入时自动创建 log_dir 目录。
        auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(
            config.log_dir + "/" + config.base_filename, config.rotate_hour,
            config.rotate_minute,
            /*truncate=*/false, /*max_files=*/config.keep_days);
        sinks.push_back(file_sink);

        // ---- 2) 创建 logger（默认异步）----
        std::shared_ptr<spdlog::logger> logger;
        if (config.async) {
            // 异步线程池：队列 8192 条，1 个后台写线程。
            // 批量推理时日志写入不阻塞推理线程。
            spdlog::init_thread_pool(8192, 1);
            logger = std::make_shared<spdlog::async_logger>(
                config.logger_name, sinks.begin(), sinks.end(),
                spdlog::thread_pool(), spdlog::async_overflow_policy::block);
            g_async = true;
        } else {
            // 【同步日志备选】如需同步（调试时日志立刻落盘、无后台线程），
            // 把 LogConfig::async 设为 false 即走这条分支：
            logger = std::make_shared<spdlog::logger>(
                config.logger_name, sinks.begin(), sinks.end());
            g_async = false;
        }

        // ---- 3) 统一日志模板 ----
        // [时间.毫秒] [线程号] [级别] [文件名:行号] 业务内容
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] [%s:%#] %v");

        // ---- 4) 级别与落盘策略 ----
        logger->set_level(config.level);
        // 遇到 warn 及以上立即 flush，保证告警类日志不丢。
        logger->flush_on(spdlog::level::warn);
        // 每 3 秒定时 flush 一次异步缓存。
        spdlog::flush_every(std::chrono::seconds(3));

        // 注册为默认 logger
        spdlog::set_default_logger(logger);
        spdlog::set_level(config.level);

        spdlog::info(
            "[log_init] logger ready. dir={}, level={}, async={}, console={}",
            config.log_dir, spdlog::level::to_string_view(config.level),
            config.async, config.enable_console);
    });
}

void setLogLevel(spdlog::level::level_enum level) {
    if (auto logger = spdlog::default_logger()) {
        logger->set_level(level);
    }
    spdlog::set_level(level);
}

void shutdownLogger() {
    // 先显式 flush 默认 logger，再 shutdown 释放异步线程池，
    // 确保异步缓存中的日志全部落盘（shutdown 本身也会 flush 已注册 logger）。
    if (auto logger = spdlog::default_logger()) {
        logger->flush();
    }

    spdlog::shutdown();
}

}  // namespace errlog
