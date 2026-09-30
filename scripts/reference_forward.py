#!/usr/bin/env python3
"""
Python reference forward pass for numerical alignment with C++ implementation.
"""

import json
import struct
import numpy as np
import math


def rmsnorm(x, weight, eps=1e-5):
    ss = np.mean(x ** 2)
    return x / np.sqrt(ss + eps) * weight


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def silu(x):
    return x / (1.0 + np.exp(-x))


def apply_rope(q, k, pos, theta=10000.0):
    """Apply RoPE to q and k in-place."""
    n_heads, head_dim = q.shape
    n_kv_heads, _ = k.shape
    for h in range(n_heads):
        for i in range(0, head_dim, 2):
            freq = 1.0 / (theta ** (i / head_dim))
            cos_val = math.cos(pos * freq)
            sin_val = math.sin(pos * freq)
            x0, x1 = q[h, i], q[h, i + 1]
            q[h, i] = x0 * cos_val - x1 * sin_val
            q[h, i + 1] = x0 * sin_val + x1 * cos_val
    for h in range(n_kv_heads):
        for i in range(0, head_dim, 2):
            freq = 1.0 / (theta ** (i / head_dim))
            cos_val = math.cos(pos * freq)
            sin_val = math.sin(pos * freq)
            x0, x1 = k[h, i], k[h, i + 1]
            k[h, i] = x0 * cos_val - x1 * sin_val
            k[h, i + 1] = x0 * sin_val + x1 * cos_val
    return q, k


def forward_token(model, pos, token_id):
    config = model["config"]
    weights = model["weights"]

    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = dim // n_heads
    hidden_dim = config["hidden_dim"]
    vocab_size = config["vocab_size"]
    rope_theta = config["rope_theta"]
    eps = config["rms_norm_eps"]

    # KV cache (kept as dict of arrays for simplicity)
    kv_cache = model.get("kv_cache", None)
    if kv_cache is None:
        kv_cache = {
            "k": np.zeros((n_layers, config["max_seq_len"], n_kv_heads, head_dim), dtype=np.float32),
            "v": np.zeros((n_layers, config["max_seq_len"], n_kv_heads, head_dim), dtype=np.float32),
        }
        model["kv_cache"] = kv_cache

    # Embedding
    x = weights["token_embedding"][token_id].copy()

    for layer in range(n_layers):
        prefix = f"layers.{layer}."
        ln1 = weights[prefix + "attention_norm"]
        wq = weights[prefix + "wq"]
        wk = weights[prefix + "wk"]
        wv = weights[prefix + "wv"]
        wo = weights[prefix + "wo"]
        ln2 = weights[prefix + "ffn_norm"]
        w_gate = weights[prefix + "w_gate"]
        w_up = weights[prefix + "w_up"]
        w_down = weights[prefix + "w_down"]

        # Attention
        h = rmsnorm(x, ln1, eps)
        q = wq @ h
        k = wk @ h
        v = wv @ h

        q = q.reshape(n_heads, head_dim)
        k = k.reshape(n_kv_heads, head_dim)
        v = v.reshape(n_kv_heads, head_dim)

        q, k = apply_rope(q, k, pos, rope_theta)

        # Write to KV cache
        kv_cache["k"][layer, pos] = k
        kv_cache["v"][layer, pos] = v

        # Attention scores
        attn_out = np.zeros((n_heads, head_dim), dtype=np.float32)
        for h_idx in range(n_heads):
            kv_head = h_idx // (n_heads // n_kv_heads)
            scores = []
            for t in range(pos + 1):
                k_t = kv_cache["k"][layer, t, kv_head]
                score = np.dot(q[h_idx], k_t) / math.sqrt(head_dim)
                scores.append(score)
            scores = np.array(scores, dtype=np.float32)
            scores = softmax(scores)
            for t in range(pos + 1):
                v_t = kv_cache["v"][layer, t, kv_head]
                attn_out[h_idx] += scores[t] * v_t

        attn_proj = wo @ attn_out.reshape(-1)
        x = x + attn_proj

        # FFN (SwiGLU)
        h = rmsnorm(x, ln2, eps)
        gate = silu(w_gate @ h)
        up = w_up @ h
        ff = w_down @ (gate * up)
        x = x + ff

    # Final norm and logits
    x = rmsnorm(x, weights["final_norm"], eps)
    logits = weights["lm_head"] @ x
    return logits


def load_model(config_path, weights_path):
    with open(config_path, "r") as f:
        json_data = json.load(f)

    config = json_data["config"]

    # Read binary weights
    weights = {}
    with open(weights_path, "rb") as f:
        for t in json_data["tensors"]:
            name = t["name"]
            shape = tuple(t["shape"])
            numel = np.prod(shape)
            raw = f.read(numel * 4)
            if len(raw) != numel * 4:
                raise ValueError(f"weights file ended while reading tensor: {name}")
            data = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            weights[name] = data.copy()
        extra = f.read(1)
        if extra:
            raise ValueError("weights file has trailing bytes")

    return {"config": config, "weights": weights}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/tiny/model.json")
    parser.add_argument("--weights", default="models/tiny/model.bin")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()

    model = load_model(args.config, args.weights)
    config = model["config"]

    # Toy tokenizer: ASCII + BOS=1
    tokens = [1] + [min(ord(c), 127) for c in args.prompt]
    print("prompt:", args.prompt)
    print("tokens:", tokens)

    # Prefill
    for i, tok in enumerate(tokens):
        logits = forward_token(model, i, tok)

    # Generate. The last prefill logits predict the first generated token.
    generated = []
    if args.n > 0:
        next_tok = int(np.argmax(logits))
        generated.append(next_tok)
        for i in range(1, args.n):
            pos = len(tokens) + i - 1
            tok = generated[-1]
            logits = forward_token(model, pos, tok)
            next_tok = int(np.argmax(logits))
            generated.append(next_tok)
            if next_tok == 2:  # EOS
                break

    print("generated tokens:", generated)
    text = "".join(chr(t) if 32 <= t < 127 else f"<{t}>" for t in generated)
    print("generated text:", text)


if __name__ == "__main__":
    main()
