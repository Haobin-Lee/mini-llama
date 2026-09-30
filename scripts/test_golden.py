#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args):
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(map(str, args))}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def main():
    binary = os.environ.get("MINI_LLAMA_BIN")
    if not binary:
        binary = str(ROOT / "build" / "mini-llama")
    binary = str(Path(binary).resolve())

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "golden"
        run_cmd([
            binary,
            "--config", "models/tiny/model.json",
            "--model", "models/tiny/model.bin",
            "--prompt", "hello",
            "--n-predict", "16",
            "--dump-logits", str(out_dir),
        ])
        run_cmd([
            sys.executable,
            "scripts/make_golden.py",
            "--config", "models/tiny/model.json",
            "--weights", "models/tiny/model.bin",
            "--prompt", "hello",
            "-n", "16",
            "--output-dir", str(out_dir),
        ])
        run_cmd([
            sys.executable,
            "scripts/compare_logits.py",
            "--cpp-dir", str(out_dir),
            "--py-dir", str(out_dir),
            "--vocab-size", "128",
        ])

    print("PASS golden_logits_alignment")


if __name__ == "__main__":
    main()
