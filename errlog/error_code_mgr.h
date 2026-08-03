// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// 错误码管理器 ErrorCodeMgr（全局单例）。
// 程序启动时加载 error_code_config.json，建立 int code / string code_str
// 到错误元数据 ErrorMeta 的双向哈希映射，供 BizLog 查表使用。

#ifndef ERRLOG_ERROR_CODE_MGR_H_
#define ERRLOG_ERROR_CODE_MGR_H_

#include <spdlog/common.h>

#include <string>
#include <unordered_map>

#include "generated/error_code_enum.h"

namespace errlog {

// 单条错误码的完整元数据。字段与 JSON 一一对应，level 预转换为 spdlog 级别。
struct ErrorMeta {
    int code = 0;                                          // 数字错误码
    std::string code_str;                                  // 语义标识
    spdlog::level::level_enum level = spdlog::level::err;  // 日志级别
    std::string msg_cn;                                    // 上层中文提示
    std::string log_desc;     // 带 {} 占位的日志描述
    bool need_alert = false;  // 是否标记 [ALERT]
};

// 全局单例：进程内只存在一份错误码表。
class ErrorCodeMgr {
public:
    // 获取单例。首次调用前必须先调用 loadFromFile()。
    static ErrorCodeMgr& instance();

    // 从 JSON 配置加载错误码表。
    void loadFromFile(const std::string& config_path);

    // 无参重载：使用 defaultConfigPath() 返回的默认路径加载。
    // 推荐入口程序直接调用此版本，避免依赖运行时工作目录。
    void loadFromFile();

    // 默认配置路径：优先环境变量 MINI_LLAMA_ERROR_CONFIG，
    // 否则用编译期注入的绝对路径 ERRLOG_DEFAULT_CONFIG_PATH，
    // 二者都缺失时回退到 "errlog/error_code_config.json"（相对工作目录）。
    static std::string defaultConfigPath();

    // 按数字 code 查询；未命中返回 nullptr（由 BizLog 处理）。
    const ErrorMeta* getMetaByCode(int code) const;

    // 按语义 code_str 查询；未命中返回 nullptr。
    const ErrorMeta* getMetaByStr(const std::string& code_str) const;

    // 便捷重载：直接用强类型枚举查询。
    const ErrorMeta* getMetaByCode(ErrorCode ec) const {
        return getMetaByCode(errorCodeToInt(ec));
    }

    // 是否已成功加载配置。
    bool loaded() const { return loaded_; }

    ErrorCodeMgr(const ErrorCodeMgr&) = delete;
    ErrorCodeMgr& operator=(const ErrorCodeMgr&) = delete;

private:
    ErrorCodeMgr() = default;

    bool loaded_ = false;
    // 双哈希映射：int code -> ErrorMeta、string code_str -> ErrorMeta。
    // by_code_ 持有实体，by_str_ 存指针以避免拷贝、且始终指向同一份数据。
    std::unordered_map<int, ErrorMeta> by_code_;
    std::unordered_map<std::string, const ErrorMeta*> by_str_;
};

}  // namespace errlog

#endif  // ERRLOG_ERROR_CODE_MGR_H_
