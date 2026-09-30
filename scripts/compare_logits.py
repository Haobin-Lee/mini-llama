#!/usr/bin/env python3
"""
Compare C++ and Python golden logits step by step.

Usage:
    python scripts/compare_logits.py \
        --cpp-dir artifacts/golden \
        --py-dir artifacts/golden \
        --vocab-size 128
"""

import argparse
import glob
import os
import re
import sys

import numpy as np


def fail(message):
    print(f"ERROR: {message}")
    sys.exit(1)


def load_logits(path, vocab_size):
    data = np.fromfile(path, dtype=np.float32)
    if data.size != vocab_size:
        fail(f"{path} has {data.size} elements, expected {vocab_size}")
    return data


def collect_step_files(directory, prefix):
    stem = "logits" if prefix == "" else f"{prefix}_logits"
    pattern = os.path.join(directory, f"{stem}_step*.bin")
    files = {}
    regex = re.compile(rf"^{re.escape(stem)}_step(\d+)\.bin$")
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        match = regex.match(basename)
        if not match:
            continue
        step = int(match.group(1))
        if step in files:
            fail(f"duplicate {prefix} logits for step {step}")
        files[step] = path
    return files


def load_token_file(path):
    if not os.path.exists(path):
        fail(f"missing token file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        return [int(part) for part in text.split()]
    except ValueError as exc:
        fail(f"invalid token file {path}: {exc}")


def compare_step(cpp_path, py_path, vocab_size, step):
    cpp = load_logits(cpp_path, vocab_size)
    py = load_logits(py_path, vocab_size)

    diff = np.abs(cpp - py)
    max_abs_error = float(np.max(diff))

    # Relative error: only consider elements where |py| >= 1e-3 to avoid
    # exploding relative error near zero (standard numerical analysis practice).
    mask = np.abs(py) >= 1e-3
    if np.any(mask):
        rel_errors = diff[mask] / np.abs(py[mask])
        max_rel_error = float(np.max(rel_errors))
    else:
        max_rel_error = 0.0

    cpp_top1 = int(np.argmax(cpp))
    py_top1 = int(np.argmax(py))
    top1_match = cpp_top1 == py_top1

    abs_ok = max_abs_error <= 1e-4
    rel_ok = max_rel_error <= 1e-4
    status = "PASS" if (abs_ok and rel_ok and top1_match) else "FAIL"

    print(f"  step {step}: {status}")
    print(f"    max_abs_error = {max_abs_error:.6e} {'OK' if abs_ok else 'FAIL'}")
    print(f"    max_rel_error = {max_rel_error:.6e} {'OK' if rel_ok else 'FAIL'} (|py| >= 1e-3)")
    print(f"    cpp top-1 = {cpp_top1}, py top-1 = {py_top1}, match = {top1_match}")

    return status == "PASS"


def compare_generation_tokens(cpp_dir, py_dir):
    cpp_path = os.path.join(cpp_dir, "generation_tokens.txt")
    py_path = os.path.join(py_dir, "python_generation_tokens.txt")
    cpp_tokens = load_token_file(cpp_path)
    py_tokens = load_token_file(py_path)
    match = cpp_tokens == py_tokens
    status = "PASS" if match else "FAIL"
    print(f"generation tokens: {status}")
    print(f"  cpp = {cpp_tokens}")
    print(f"  py  = {py_tokens}")
    return match


def main():
    parser = argparse.ArgumentParser(description="Compare C++ and Python logits")
    parser.add_argument("--cpp-dir", required=True, help="directory with cpp_logits_step*.bin")
    parser.add_argument("--py-dir", required=True, help="directory with python_logits_step*.bin")
    parser.add_argument("--vocab-size", type=int, default=128)
    args = parser.parse_args()

    cpp_files = collect_step_files(args.cpp_dir, "")
    py_files = collect_step_files(args.py_dir, "python")

    if not cpp_files:
        fail(f"no logits_step*.bin found in {args.cpp_dir}")
    if not py_files:
        fail(f"no python_logits_step*.bin found in {args.py_dir}")

    cpp_steps = set(cpp_files.keys())
    py_steps = set(py_files.keys())
    if cpp_steps != py_steps:
        missing_py = sorted(cpp_steps - py_steps)
        missing_cpp = sorted(py_steps - cpp_steps)
        if missing_py:
            print(f"ERROR: missing Python logits for steps: {missing_py}")
        if missing_cpp:
            print(f"ERROR: missing C++ logits for steps: {missing_cpp}")
        sys.exit(1)

    all_pass = True
    print(f"Comparing {len(cpp_steps)} steps...")
    print("")

    for step in sorted(cpp_steps):
        ok = compare_step(cpp_files[step], py_files[step], args.vocab_size, step)
        if not ok:
            all_pass = False

    if not compare_generation_tokens(args.cpp_dir, args.py_dir):
        all_pass = False

    print("")
    if all_pass:
        print("RESULT: ALL PASSED")
        print("  abs_error <= 1e-4")
        print("  relative_error <= 1e-4")
        print("  top-1 token match")
    else:
        print("RESULT: SOME STEPS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
