// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_TOKENIZER_H_
#define INCLUDE_MINI_LLAMA_TOKENIZER_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_llama {

// Tokenizer interface: text <-> token ids.
class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decodeToken(int token) const = 0;
    virtual std::string decode(const std::vector<int>& tokens) const = 0;

    virtual int vocabSize() const = 0;
    virtual int bosId() const = 0;
    virtual int eosId() const = 0;
    virtual int unkId() const = 0;
};

// Character-level ASCII tokenizer for ids 0..127.
class AsciiTokenizer : public ITokenizer {
public:
    static constexpr int kVocabSize = 128;

    AsciiTokenizer() = default;

    std::vector<int> encode(const std::string& text) const override;
    std::string decodeToken(int token) const override;
    std::string decode(const std::vector<int>& tokens) const override;

    int vocabSize() const override { return kVocabSize; }
    int bosId() const override { return 1; }
    int eosId() const override { return 2; }
    int unkId() const override { return 0; }
};

// Vocabulary loaded from a JSON file.
// Format: [{"id": 0, "content": "<unk>", "special": true}, ...]
// Encode: greedy longest-match, UNK fallback.
class JsonVocabTokenizer : public ITokenizer {
public:
    explicit JsonVocabTokenizer(const std::string& vocab_path);

    std::vector<int> encode(const std::string& text) const override;
    std::string decodeToken(int token) const override;
    std::string decode(const std::vector<int>& tokens) const override;

    int vocabSize() const override { return vocab_size_; }
    int bosId() const override { return bos_id_; }
    int eosId() const override { return eos_id_; }
    int unkId() const override { return unk_id_; }

    // True only if the vocab file parsed successfully in the constructor.
    bool valid() const { return valid_; }

private:
    struct VocabEntry {
        int id = 0;
        std::string content;
        bool special = false;
    };

    bool valid_ = false;
    int vocab_size_ = 0;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int unk_id_ = 0;

    std::vector<VocabEntry> id_to_entry_;
    std::vector<std::pair<std::string, int>> content_to_id_;
};

// ---------------------------------------------------------------------------
// BPE tokenizer (GPT-2 style).
// Loads vocab.json + merges.txt + special_tokens.json.
// ---------------------------------------------------------------------------
class BpeTokenizer : public ITokenizer {
public:
    BpeTokenizer() = default;
    ~BpeTokenizer() override = default;

    bool Load(const std::string& vocab_path, const std::string& merges_path,
              const std::string& special_path);

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
    bool loadVocab(const std::string& vocab_path);
    bool loadMerges(const std::string& merges_path);
    bool loadSpecialTokens(const std::string& special_path);

private:
    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;

    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int> merge_ranks_;

    // Special tokens (e.g. <|im_start|>) sorted by descending length for
    // longest-match.
    std::vector<std::pair<std::string, int>> special_tokens_;

    std::vector<std::string> b2u_;                  // byte -> unicode string
    std::unordered_map<std::string, uint8_t> u2b_;  // unicode string -> byte
};

// Factory:Tokenizer if vocab_pathTokenizer if vocab_path exists, else
// AsciiTokenizer.
std::unique_ptr<ITokenizer> createTokenizer(const std::string& vocab_path);

// Load BPE tokenizer from vocab.json + merges.txt + special_tokens.json.
std::unique_ptr<ITokenizer> createBpeTokenizer(const std::string& vocab_path,
                                               const std::string& merges_path,
                                               const std::string& special_path);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_TOKENIZER_H_
