// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
#define INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// GgufTokenizer: loads tokenizer data directly from GGUF metadata.
//
// Reads:
//   - tokenizer.ggml.tokens      (vocab pieces)
//   - tokenizer.ggml.merges      (BPE merge rules)
//   - tokenizer.ggml.token_type  (to identify special/control tokens)
//   - tokenizer.ggml.bos_token_id
//   - tokenizer.ggml.eos_token_id
//
// Implements GPT-2 style BPE with byte-level fallback.
// ---------------------------------------------------------------------------
class GgufTokenizer : public ITokenizer {
public:
    GgufTokenizer() = default;
    ~GgufTokenizer() override = default;

    // Load tokenizer from a GGUF file. Returns false if the file lacks
    // the required tokenizer metadata.
    bool load(const std::string& gguf_path);

    std::vector<int> encode(const std::string& text) const override;
    std::string decodeToken(int token) const override;
    std::string decode(const std::vector<int>& tokens) const override;

    int vocabSize() const override {
        return static_cast<int>(id_to_token_.size());
    }
    int bosId() const override { return bos_id_; }
    int eosId() const override { return eos_id_; }
    int unkId() const override { return unk_id_; }

private:
    void buildByteMappings();

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int> merge_ranks_;
    std::vector<std::pair<std::string, int>> special_tokens_;

    // Byte <-> unicode mappings (GPT-2 style)
    std::vector<std::string> b2u_;
    std::unordered_map<std::string, uint8_t> u2b_;

    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;
};

// ---------------------------------------------------------------------------
// Factory: create a GgufTokenizer from a GGUF file.
// Returns nullptr if the GGUF lacks tokenizer metadata.
// ---------------------------------------------------------------------------
std::unique_ptr<ITokenizer> createGgufTokenizer(const std::string& gguf_path);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
