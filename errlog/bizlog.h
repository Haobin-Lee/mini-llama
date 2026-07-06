// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// 业务日志工具类 BizLog。
//
// 只需传入 ErrorCode（或数字 code）+ 上下文参数，即可自动：
//   1) 查表匹配日志级别、msg_cn、log_desc 模板；
//   2) 用可变参数填充 log_desc 中的 {} 占位符；
//   3) 输出统一格式：数字 code + 语义 code_str + 中文 msg + detail 明细；
//   4) need_alert=true 追加 [ALERT] 标记；critical 级附带堆栈；
//   5) 未知错误码兜底为 error 日志，绝不崩溃。
//
// 提供两套“有/无 session_id”重载，无 session 场景直接调用、无需传空串。
// 由于要携带调用处文件名/行号，对外统一用宏 BIZLOG / BIZLOG_S 调用。

#ifndef ERRLOG_BIZLOG_H_
#define ERRLOG_BIZLOG_H_

#include <spdlog/spdlog.h>

#include <string>
#include <utility>

#include "errlog/error_code_mgr.h"
#include "generated/error_code_enum.h"

namespace errlog {

class BizLog {
public:
    // ============ 带 session_id 的重载（多轮对话场景） ============

    // 枚举版 + session。
    template <typename... Args>
    static void print(spdlog::source_loc loc, ErrorCode ec,
                      const std::string& session_id, Args&&... args) {
        const ErrorMeta* meta =
            ErrorCodeMgr::instance().getMetaByCode(errorCodeToInt(ec));
        emit(loc, meta, errorCodeToInt(ec), &session_id,
             std::forward<Args>(args)...);
    }

    // 数字 code 版 + session（兼容历史数字错误码）。
    template <typename... Args>
    static void print(spdlog::source_loc loc, int code,
                      const std::string& session_id, Args&&... args) {
        const ErrorMeta* meta = ErrorCodeMgr::instance().getMetaByCode(code);
        emit(loc, meta, code, &session_id, std::forward<Args>(args)...);
    }

    // 枚举版，无 session。
    template <typename... Args>
    static void print(spdlog::source_loc loc, ErrorCode ec, Args&&... args) {
        const ErrorMeta* meta =
            ErrorCodeMgr::instance().getMetaByCode(errorCodeToInt(ec));
        emit(loc, meta, errorCodeToInt(ec), /*session=*/nullptr,
             std::forward<Args>(args)...);
    }

    // 数字 code 版，无 session。
    template <typename... Args>
    static void print(spdlog::source_loc loc, int code, Args&&... args) {
        const ErrorMeta* meta = ErrorCodeMgr::instance().getMetaByCode(code);
        emit(loc, meta, code, /*session=*/nullptr, std::forward<Args>(args)...);
    }

private:
    // 统一出口：拼装最终日志文本并按对应级别落盘。
    // meta==nullptr 表示未知错误码，走兜底逻辑。
    // session==nullptr 表示无 session 重载，日志完全不输出 session 字段。
    template <typename... Args>
    static void emit(spdlog::source_loc loc, const ErrorMeta* meta, int code,
                     const std::string* session, Args&&... args) {
        // ---- 兜底：未知错误码 ----
        if (meta == nullptr) {
            std::string detail = safeFormat("(unknown code, {} arg(s) dropped)",
                                            sizeof...(Args));
            std::string body = buildBody(session, code, "UNKNOWN", "未知错误码",
                                         detail, /*alert=*/true);
            spdlog::default_logger_raw()->log(loc, spdlog::level::err, body);
            return;
        }

        // ---- 正常：填充 log_desc 占位符 ----
        std::string detail =
            safeFormat(meta->log_desc, std::forward<Args>(args)...);
        std::string body = buildBody(session, meta->code, meta->code_str,
                                     meta->msg_cn, detail, meta->need_alert);

        // 按查表得到的级别输出（携带调用处 file:line）。
        spdlog::default_logger_raw()->log(loc, meta->level, body);

        // critical 级自动附带堆栈信息（此处输出调用点定位，
        // 生产环境可替换为 backtrace/execinfo 的完整调用栈）。
        if (meta->level == spdlog::level::critical) {
            spdlog::default_logger_raw()->log(
                loc, spdlog::level::critical,
                safeFormat("[STACKTRACE] at {}:{} ({})", loc.filename, loc.line,
                           loc.funcname ? loc.funcname : "?"));
        }
    }

    // 组装统一日志正文：
    // {[ALERT] }[session:xxx ]<[code:..][code_str:..][msg:..]> detail:...
    static std::string buildBody(const std::string* session, int code,
                                 const std::string& code_str,
                                 const std::string& msg_cn,
                                 const std::string& detail, bool need_alert) {
        std::string s;
        if (need_alert) {
            s += "[ALERT] ";  // 严重异常本地标记
        }
        // 有 session 才输出 [session:xxx]，无 session 重载完全省略该字段。
        if (session != nullptr) {
            s += "[session:" + *session + "] ";
        }
        s += "[code:" + std::to_string(code) + "]";
        s += "[code_str:" + code_str + "]";
        s += "[msg:" + msg_cn + "]";
        s += " detail:" + detail;
        return s;
    }

    // 安全格式化：占位符与参数个数不匹配时不抛异常、不崩溃，
    // 而是回退成“原始模板 + 附注”，保证日志系统自身永不成为故障点。
    template <typename... Args>
    static std::string safeFormat(const std::string& fmt_str, Args&&... args) {
        try {
            return fmt::vformat(fmt_str, fmt::make_format_args(args...));
        } catch (const std::exception& e) {
            return fmt_str + " [fmt-error: " + e.what() + "]";
        }
    }
};

// ============ 对外调用宏（自动捕获调用处 file:line:func） ============
//
// 无 session：   BIZLOG(ErrorCode::kXxx, arg1, arg2, ...);
// 带 session：   BIZLOG_S(ErrorCode::kXxx, session_id, arg1, arg2, ...);
// 数字 code 同样适用（第一个参数传 int 即可）。
//
// 使用 __FILE__/__LINE__/__FUNCTION__ 定位到“调用点”，

#define BIZLOG(ec, ...)                                                \
    ::errlog::BizLog::print(                                           \
        spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, (ec), \
        ##__VA_ARGS__)

#define BIZLOG_S(ec, session_id, ...)                                  \
    ::errlog::BizLog::print(                                           \
        spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, (ec), \
        (session_id), ##__VA_ARGS__)

}  // namespace errlog

#endif  // ERRLOG_BIZLOG_H_
