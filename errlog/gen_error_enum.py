#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_error_enum.py

读取 error_code_config.json，自动生成强类型 C++ 枚举头文件
generated/error_code_enum.h。

生成规则：
  * enum class ErrorCode : int
  * 枚举成员名 = 'k' + code_str   (例如 code_str="TensorShapeEmpty" -> kTensorShapeEmpty)
  * 枚举数值   = code
  * 附带 ErrorCodeToInt / IntToErrorCode 转换工具函数
  * 文件头部标注“自动生成，禁止手动修改”

加载阶段强校验：
  * code 全局唯一
  * code_str 全局唯一
  * 生成出来的枚举成员名全局唯一
  * 关键字段缺失
一旦发现重复/缺失，打印 [CRITICAL] 日志并以非 0 退出码终止（对应需求“重复则终止程序”）。

用法：
  python3 gen_error_enum.py \
      --config error_code_config.json \
      --output generated/error_code_enum.h
"""

import argparse
import json
import os
import sys

# 允许出现在配置里的日志级别（与 spdlog 级别名保持一致），仅做合法性校验。
VALID_LEVELS = {"trace", "debug", "info", "warn", "err", "critical"}


def fatal(msg: str) -> "None":
    """打印 CRITICAL 日志并终止程序（模拟 spdlog critical 行为）。"""
    print("[CRITICAL][gen_error_enum] " + msg, file=sys.stderr)
    sys.exit(1)


def load_config(path: str) -> dict:
    """读取并解析 JSON 配置，文件缺失或格式错误直接终止。"""
    if not os.path.isfile(path):
        fatal("config file not found: {}".format(path))
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        fatal("config json parse error: {}".format(e))


def enum_name(code_str: str) -> str:
    """由 code_str 生成枚举成员名：加 'k' 前缀。

    配置里的 code_str 已经是 CamelCase（不带 k），此处只补前缀，
    保证枚举名与 code_str 一一对应、可反推。
    """
    if not code_str:
        fatal("empty code_str encountered")
    return "k" + code_str


def validate_and_collect(cfg: dict) -> list:
    """校验配置并返回规整后的错误条目列表。

    强校验项：字段完整性、level 合法、code/code_str/枚举名全局唯一。
    """
    errors = cfg.get("errors")
    if not isinstance(errors, list) or not errors:
        fatal("config must contain a non-empty 'errors' array")

    seen_code = {}       # code -> code_str，用于查重时给出冲突详情
    seen_code_str = set()
    seen_enum = set()
    collected = []

    for idx, item in enumerate(errors):
        # 1) 必填字段校验
        for field in ("code", "code_str", "level", "msg_cn", "log_desc",
                      "need_alert"):
            if field not in item:
                fatal("errors[{}] missing required field '{}'".format(idx, field))

        code = item["code"]
        code_str = item["code_str"]
        level = item["level"]

        # 2) 类型/取值校验
        if not isinstance(code, int):
            fatal("errors[{}].code must be int, got {!r}".format(idx, code))
        if not isinstance(code_str, str) or not code_str:
            fatal("errors[{}].code_str must be non-empty string".format(idx))
        if level not in VALID_LEVELS:
            fatal("errors[{}].level '{}' invalid, must be one of {}"
                  .format(idx, level, sorted(VALID_LEVELS)))
        if not isinstance(item["need_alert"], bool):
            fatal("errors[{}].need_alert must be bool".format(idx))

        name = enum_name(code_str)

        # 3) 全局唯一性校验（code / code_str / 枚举名）
        if code in seen_code:
            fatal("duplicate code {} (code_str='{}' and '{}')"
                  .format(code, seen_code[code], code_str))
        if code_str in seen_code_str:
            fatal("duplicate code_str '{}'".format(code_str))
        if name in seen_enum:
            fatal("duplicate enum name '{}'".format(name))

        seen_code[code] = code_str
        seen_code_str.add(code_str)
        seen_enum.add(name)
        collected.append((name, code, code_str))

    return collected


def render_header(entries: list, config_path: str) -> str:
    """渲染最终的 C++ 头文件文本。"""
    lines = []
    lines.append("// clang-format off")
    lines.append("// =============================================================")
    lines.append("//  AUTO-GENERATED FILE - DO NOT EDIT BY HAND")
    lines.append("//  本文件由 gen_error_enum.py 依据 {} 自动生成。".format(
        os.path.basename(config_path)))
    lines.append("//  如需增删错误码，请修改 JSON 配置后重新运行生成脚本。")
    lines.append("// =============================================================")
    lines.append("")
    lines.append("#ifndef ERRLOG_GENERATED_ERROR_CODE_ENUM_H_")
    lines.append("#define ERRLOG_GENERATED_ERROR_CODE_ENUM_H_")
    lines.append("")
    lines.append("namespace mini_llama {")
    lines.append("namespace errlog {")
    lines.append("")
    lines.append("// 强类型错误码枚举：成员名 = 'k' + code_str，数值 = code。")
    lines.append("enum class ErrorCode : int {")
    for name, code, code_str in entries:
        lines.append("    {} = {},  // {}".format(name, code, code_str))
    lines.append("};")
    lines.append("")
    lines.append("// ErrorCode -> int 转换。")
    lines.append("inline int errorCodeToInt(ErrorCode ec) {")
    lines.append("    return static_cast<int>(ec);")
    lines.append("}")
    lines.append("")
    lines.append("// int -> ErrorCode 转换（不校验合法性，非法值的查表由 "
                 "ErrorCodeMgr 兜底）。")
    lines.append("inline ErrorCode intToErrorCode(int code) {")
    lines.append("    return static_cast<ErrorCode>(code);")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace errlog")
    lines.append("}  // namespace mini_llama")
    lines.append("")
    lines.append("#endif  // ERRLOG_GENERATED_ERROR_CODE_ENUM_H_")
    lines.append("// clang-format on")
    lines.append("")
    return "\n".join(lines)


def main() -> "None":
    parser = argparse.ArgumentParser(
        description="Generate C++ ErrorCode enum from JSON config.")
    parser.add_argument("--config", required=True,
                        help="path to error_code_config.json")
    parser.add_argument("--output", required=True,
                        help="path to generated error_code_enum.h")
    args = parser.parse_args()

    cfg = load_config(args.config)
    entries = validate_and_collect(cfg)
    header_text = render_header(entries, args.config)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(header_text)

    print("[INFO][gen_error_enum] generated {} error codes -> {}"
          .format(len(entries), args.output))


if __name__ == "__main__":
    main()
