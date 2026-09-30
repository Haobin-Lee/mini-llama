#!/usr/bin/env python3
"""
Generate golden logits and token sequence from Python reference implementation.

Usage:
    python scripts/make_golden.py \
        --config models/tiny/model.json \
        --weights models/tiny/model.bin \
        --prompt "hello" \
        -n 16 \
        --output-dir artifacts/golden
"""

import argparse
import json
import os
import sys

import numpy as np

# Import reference forward from the same scripts directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_forward import load_model, forward_token


def main():
    parser = argparse.ArgumentParser(description="Generate golden logits from Python reference")
    parser.add_argument("--config", default="models/tiny/model.json")
    parser.add_argument("--weights", default="models/tiny/model.bin")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("-n", type=int, default=16, help="number of tokens to generate")
    parser.add_argument("--output-dir", default="artifacts/golden", help="output directory")
    args = parser.parse_args()
    if args.n < 0:
        parser.error("-n must be non-negative")

    model = load_model(args.config, args.weights)
    config = model["config"]

    # Toy tokenizer: ASCII + BOS=1
    tokens = [1] + [min(ord(c), 127) for c in args.prompt]
    if len(tokens) > config["max_seq_len"]:
        parser.error(f"prompt has {len(tokens)} tokens, max_seq_len is {config['max_seq_len']}")
    if len(tokens) + args.n > config["max_seq_len"]:
        parser.error(
            f"prompt tokens ({len(tokens)}) + n ({args.n}) exceeds "
            f"max_seq_len ({config['max_seq_len']})"
        )
    print("prompt:", args.prompt)
    print("tokens:", tokens)

    os.makedirs(args.output_dir, exist_ok=True)

    step = 0
    # Prefill
    for i, tok in enumerate(tokens):
        logits = forward_token(model, i, tok)
        path = os.path.join(args.output_dir, f"python_logits_step{step}.bin")
        logits.astype(np.float32).tofile(path)
        print(f"  step {step}: prefill pos={i}, saved {path}")
        step += 1

    # Generate
    generated = []
    if args.n > 0:
        next_tok = int(np.argmax(logits))
        generated.append(next_tok)
        for i in range(1, args.n):
            pos = len(tokens) + i - 1
            tok = generated[-1]
            logits = forward_token(model, pos, tok)
            path = os.path.join(args.output_dir, f"python_logits_step{step}.bin")
            logits.astype(np.float32).tofile(path)
            print(f"  step {step}: decode pos={pos}, saved {path}")
            step += 1
            next_tok = int(np.argmax(logits))
            generated.append(next_tok)
            if next_tok == 2:  # EOS
                break

    # Save generation tokens
    gen_path = os.path.join(args.output_dir, "python_generation_tokens.txt")
    with open(gen_path, "w") as f:
        f.write(" ".join(str(t) for t in generated) + "\n")
    print(f"generation tokens: {generated}")
    print(f"saved to {gen_path}")


if __name__ == "__main__":
    main()
