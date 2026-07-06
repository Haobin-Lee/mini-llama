// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "log_guard.h"

namespace errlog {
LogGuard::LogGuard() {
    // 初始化日志系统
    errlog::initLogger();
    errlog::ErrorCodeMgr::instance().loadFromFile();
};
LogGuard::~LogGuard() {
    // 销毁日志系统
    errlog::shutdownLogger();
};
}  // namespace errlog
