// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// spdlog 日志初始化模块。
// 负责创建 ./llama_logs 目录下按天切割的日志文件 + 控制台彩色输出，
// 默认使用异步日志（批量推理不阻塞推理线程）。

#ifndef ERRLOG_LOG_INIT_H_
#define ERRLOG_LOG_INIT_H_

#include <spdlog/common.h>

#include <string>

namespace errlog {

// 日志初始化参数（均有合理默认值）。
struct LogConfig {
    std::string log_dir = "./llama_logs";          // 日志目录，自动创建
    std::string logger_name = "mini_llama";        // 全局 logger 名
    std::string base_filename = "mini_llama.log";  // daily sink 基础文件名
    int keep_days = 30;                            // 保留天数（滚动清理）
    int rotate_hour = 0;                           // 每天切割时刻（时）
    int rotate_minute = 0;                         // 每天切割时刻（分）

    // 默认级别 info；开发模式可设为 spdlog::level::debug。
    spdlog::level::level_enum level = spdlog::level::info;

    // 是否输出到控制台。设为 false 即可关闭控制台彩色打印，仅写文件。
    bool enable_console = true;

    // 是否使用异步日志。false 则退化为同步日志（见 .cc 中的备选注释）。
    bool async = false;
};

// 初始化全局 logger 并设为 spdlog 默认 logger。
// 重复调用是安全的（内部保证只初始化一次）。
void initLogger(const LogConfig& config = LogConfig{});

// 运行期切换日志级别（例如开发模式切 debug）。
void setLogLevel(spdlog::level::level_enum level);

// 程序退出前调用：flush 并释放 sink，确保异步缓存日志全部落盘。
void shutdownLogger();

}  // namespace errlog

#endif  // ERRLOG_LOG_INIT_H_
