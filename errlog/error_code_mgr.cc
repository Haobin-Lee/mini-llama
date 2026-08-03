// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "errlog/error_code_mgr.h"

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#if defined(__linux__)
#include <limits.h>
#include <unistd.h>
#endif

namespace errlog {

namespace {

// 返回当前可执行程序所在目录（Linux 通过 /proc/self/exe 解析）。
// 解析失败返回空串。
std::string executableDir() {
#if defined(__linux__)
    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return std::filesystem::path(buf).parent_path().string();
    }
#endif
    return std::string();
}

// 将配置里的 level 字符串转换为 spdlog 级别枚举。
// 非法级别直接视为致命错误终止（配置阶段就应暴露问题）。
spdlog::level::level_enum parseLevel(const std::string& level,
                                     const std::string& code_str) {
    if (level == "trace") {
        return spdlog::level::trace;
    }
    if (level == "debug") {
        return spdlog::level::debug;
    }
    if (level == "info") {
        return spdlog::level::info;
    }
    if (level == "warn") {
        return spdlog::level::warn;
    }
    if (level == "err") {
        return spdlog::level::err;
    }
    if (level == "critical") {
        return spdlog::level::critical;
    }

    // 这里 spdlog 可能尚未初始化 sink，用 critical + 退出保证问题不被吞掉。
    spdlog::critical(
        "[ErrorCodeMgr] invalid level '{}' for code_str='{}', abort", level,
        code_str);
    std::exit(EXIT_FAILURE);
}

}  // namespace

ErrorCodeMgr& ErrorCodeMgr::instance() {
    // C++11 起局部静态变量初始化线程安全（Magic Static）。
    static ErrorCodeMgr instance;
    return instance;
}

std::string ErrorCodeMgr::defaultConfigPath() {
    // 1) 环境变量优先，便于部署时灵活指定配置位置。
    if (const char* env = std::getenv("MINI_LLAMA_ERROR_CONFIG")) {
        if (env[0] != '\0') {
            return env;
        }
    }
    // 2) 可执行程序同级的 errlog/ 目录（编译时 POST_BUILD 已拷贝到此），
    //    不依赖运行时工作目录，程序放到哪里都能就近找到配置。
    const std::string exe_dir = executableDir();
    if (!exe_dir.empty()) {
        std::filesystem::path p = std::filesystem::path(exe_dir) / "errlog" /
                                  "error_code_config.json";
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) {
            return p.string();
        }
    }
    // 3) 编译期由 CMake 注入的源码配置绝对路径（开发态兜底）。
#ifdef ERRLOG_DEFAULT_CONFIG_PATH
    return ERRLOG_DEFAULT_CONFIG_PATH;
#else
    // 4) 最终兜底：相对当前工作目录（需在项目根目录运行）。
    return "errlog/error_code_config.json";
#endif
}

void ErrorCodeMgr::loadFromFile() { loadFromFile(defaultConfigPath()); }

void ErrorCodeMgr::loadFromFile(const std::string& config_path) {
    // 1) 打开文件，缺失即致命退出。
    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        spdlog::critical("[ErrorCodeMgr] cannot open config file: {}",
                         config_path);
        std::exit(EXIT_FAILURE);
    }

    // 2) 解析 JSON，格式错误即致命退出。
    nlohmann::json root;
    try {
        ifs >> root;
    } catch (const nlohmann::json::exception& e) {
        spdlog::critical("[ErrorCodeMgr] json parse error in {}: {}",
                         config_path, e.what());
        std::exit(EXIT_FAILURE);
    }

    if (!root.contains("errors") || !root["errors"].is_array()) {
        spdlog::critical(
            "[ErrorCodeMgr] config must contain an 'errors' array: {}",
            config_path);
        std::exit(EXIT_FAILURE);
    }

    by_code_.clear();
    by_str_.clear();

    // 3) 逐条读取并建立双映射，同时做全局唯一性校验。
    for (const auto& item : root["errors"]) {
        ErrorMeta meta;
        try {
            meta.code = item.at("code").get<int>();
            meta.code_str = item.at("code_str").get<std::string>();
            meta.level =
                parseLevel(item.at("level").get<std::string>(), meta.code_str);
            meta.msg_cn = item.at("msg_cn").get<std::string>();
            meta.log_desc = item.at("log_desc").get<std::string>();
            meta.need_alert = item.at("need_alert").get<bool>();
        } catch (const nlohmann::json::exception& e) {
            spdlog::critical("[ErrorCodeMgr] bad error entry in {}: {}",
                             config_path, e.what());
            std::exit(EXIT_FAILURE);
        }

        // code 全局唯一。
        if (by_code_.count(meta.code) != 0) {
            spdlog::critical(
                "[ErrorCodeMgr] duplicate code {} (code_str='{}'), abort",
                meta.code, meta.code_str);
            std::exit(EXIT_FAILURE);
        }
        // code_str 全局唯一。
        if (by_str_.count(meta.code_str) != 0) {
            spdlog::critical(
                "[ErrorCodeMgr] duplicate code_str '{}' (code={}), abort",
                meta.code_str, meta.code);
            std::exit(EXIT_FAILURE);
        }

        auto res = by_code_.emplace(meta.code, std::move(meta));
        // by_str_ 指向 by_code_ 中真正存储的那份实体。
        const ErrorMeta& stored = res.first->second;
        by_str_.emplace(stored.code_str, &stored);
    }

    loaded_ = true;
    spdlog::error("[ErrorCodeMgr] loaded {} error codes from {}",
                  by_code_.size(), config_path);
}

const ErrorMeta* ErrorCodeMgr::getMetaByCode(int code) const {
    auto it = by_code_.find(code);
    return it == by_code_.end() ? nullptr : &it->second;
}

const ErrorMeta* ErrorCodeMgr::getMetaByStr(const std::string& code_str) const {
    auto it = by_str_.find(code_str);
    return it == by_str_.end() ? nullptr : it->second;
}

}  // namespace errlog
