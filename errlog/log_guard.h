// Copyright (c) 2026
// SPDX-License-Identifier: MIT

// RAII日志启动器

#include "error_code_mgr.h"
#include "log_init.h"

namespace errlog {

class LogGuard {
public:
    LogGuard();
    ~LogGuard();

    LogGuard(const LogGuard&) = delete;
    LogGuard& operator=(const LogGuard&) = delete;
    LogGuard(LogGuard&&) = delete;
    LogGuard& operator=(LogGuard&&) = delete;
};

}  // namespace errlog
