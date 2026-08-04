// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mini_llama/tokenizer.h"
#include "tests/test_main.h"

namespace {

struct TempBpeFiles {
    std::filesystem::path dir;
    std::filesystem::path vocab;
    std::filesystem::path merges;
    std::filesystem::path special;
};

static TempBpeFiles writeTempBpeFiles() {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    TempBpeFiles files;
    files.dir = std::filesystem::temp_directory_path() /
                ("mini_llama_bpe_tokenizer_" + std::to_string(stamp));
    std::filesystem::create_directories(files.dir);
    files.vocab = files.dir / "vocab.json";
    files.merges = files.dir / "merges.txt";
    files.special = files.dir / "special_tokens.json";

    {
        std::ofstream out(files.vocab);
        out << "{\n"
            << "  \"h\": 0,\n"
            << "  \"e\": 1,\n"
            << "  \"l\": 2,\n"
            << "  \"o\": 3,\n"
            << "  \"he\": 4,\n"
            << "  \"hel\": 5,\n"
            << "  \"hell\": 6,\n"
            << "  \"hello\": 7,\n"
            << "  \"<|im_start|>\": 8,\n"
            << "  \"<unk>\": 9,\n"
            << "  \"<bos>\": 10,\n"
            << "  \"<eos>\": 11\n"
            << "}\n";
    }
    {
        std::ofstream out(files.merges);
        out << "h e\n"
            << "he l\n"
            << "hel l\n"
            << "hell o\n";
    }
    {
        std::ofstream out(files.special);
        out << "{ \"bos_id\": 10, \"eos_id\": 11, \"unk_id\": 9 }\n";
    }
    return files;
}

static void removeTempBpeFiles(const TempBpeFiles& files) {
    std::error_code ec;
    std::filesystem::remove_all(files.dir, ec);
}

}  // namespace

// ---------------------------------------------------------------------------
// mini_llama::AsciiTokenizer
// ---------------------------------------------------------------------------
static bool testAsciiEncodeBasic() {
    mini_llama::AsciiTokenizer tok;
    auto tokens = tok.encode("hi");
    MINI_LLAMA_ASSERT_EQ(tokens.size(), 3);
    MINI_LLAMA_ASSERT_EQ(tokens[0], tok.bosId());  // 1
    MINI_LLAMA_ASSERT_EQ(tokens[1], static_cast<int>('h'));
    MINI_LLAMA_ASSERT_EQ(tokens[2], static_cast<int>('i'));
    return true;
}

static bool testAsciiEncodeEmpty() {
    mini_llama::AsciiTokenizer tok;
    auto tokens = tok.encode("");
    MINI_LLAMA_ASSERT_EQ(tokens.size(), 1);
    MINI_LLAMA_ASSERT_EQ(tokens[0], tok.bosId());
    return true;
}

static bool testAsciiDecodeToken() {
    mini_llama::AsciiTokenizer tok;
    MINI_LLAMA_ASSERT_TRUE(tok.decodeToken(tok.bosId()) == "<bos>");
    MINI_LLAMA_ASSERT_TRUE(tok.decodeToken(tok.eosId()) == "<eos>");
    MINI_LLAMA_ASSERT_TRUE(tok.decodeToken(tok.unkId()) == "<unk>");
    MINI_LLAMA_ASSERT_TRUE(tok.decodeToken(static_cast<int>('a')) == "a");
    MINI_LLAMA_ASSERT_TRUE(tok.decodeToken(32) == " ");
    MINI_LLAMA_ASSERT_TRUE(tok.decodeToken(10) == "");        // control char
    MINI_LLAMA_ASSERT_TRUE(tok.decodeToken(127) == "");       // DEL
    MINI_LLAMA_ASSERT_TRUE(tok.decodeToken(200) == "<unk>");  // out of range
    return true;
}

static bool testAsciiDecodeSequence() {
    mini_llama::AsciiTokenizer tok;
    std::vector<int> tokens = {tok.bosId(), static_cast<int>('h'),
                               static_cast<int>('i')};
    std::string text = tok.decode(tokens);
    MINI_LLAMA_ASSERT_TRUE(text == "<bos>hi");
    return true;
}

static bool testAsciiVocabSize() {
    mini_llama::AsciiTokenizer tok;
    MINI_LLAMA_ASSERT_EQ(tok.vocabSize(), 128);
    MINI_LLAMA_ASSERT_EQ(tok.bosId(), 1);
    MINI_LLAMA_ASSERT_EQ(tok.eosId(), 2);
    MINI_LLAMA_ASSERT_EQ(tok.unkId(), 0);
    return true;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
static bool testFactoryUsesJsonWhenExists() {
    auto tok = mini_llama::createTokenizer("models/tiny/vocab.json");
    MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
    MINI_LLAMA_ASSERT_EQ(tok->vocabSize(), 128);
    return true;
}

static bool testFactoryFallsBackToAscii() {
    auto tok = mini_llama::createTokenizer("");
    MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
    MINI_LLAMA_ASSERT_EQ(tok->vocabSize(), 128);  //
    return true;
}

static bool testFactoryFallsBackWhenMissing() {
    auto tok = mini_llama::createTokenizer("models/tiny/does_not_exist.json");
    MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
    MINI_LLAMA_ASSERT_EQ(tok->vocabSize(),
                         128);  // mini_llama::AsciiTokenizer fallback
    return true;
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------
static bool testBpeEncodeDecodeAndSpecialToken() {
    TempBpeFiles files = writeTempBpeFiles();
    auto tok = mini_llama::createBpeTokenizer(
        files.vocab.string(), files.merges.string(), files.special.string());
    MINI_LLAMA_ASSERT_TRUE(tok != nullptr);
    MINI_LLAMA_ASSERT_EQ(tok->bosId(), 10);
    MINI_LLAMA_ASSERT_EQ(tok->eosId(), 11);
    MINI_LLAMA_ASSERT_EQ(tok->unkId(), 9);

    std::vector<int> hello = tok->encode("hello");
    MINI_LLAMA_ASSERT_EQ(hello.size(), 1u);
    MINI_LLAMA_ASSERT_EQ(hello[0], 7);
    MINI_LLAMA_ASSERT_TRUE(tok->decode(hello) == "hello");

    std::vector<int> special = tok->encode("<|im_start|>");
    MINI_LLAMA_ASSERT_EQ(special.size(), 1u);
    MINI_LLAMA_ASSERT_EQ(special[0], 8);
    MINI_LLAMA_ASSERT_TRUE(tok->decode(special) == "<|im_start|>");

    removeTempBpeFiles(files);
    return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static struct tokenizerTestRegistrar {
    tokenizerTestRegistrar() {
        registerTest("ascii_encode_basic", testAsciiEncodeBasic);
        registerTest("ascii_encode_empty", testAsciiEncodeEmpty);
        registerTest("ascii_decode_token", testAsciiDecodeToken);
        registerTest("ascii_decode_sequence", testAsciiDecodeSequence);
        registerTest("ascii_vocab_size", testAsciiVocabSize);
        registerTest("factory_uses_json_when_exists",
                     testFactoryUsesJsonWhenExists);
        registerTest("factory_falls_back_to_ascii",
                     testFactoryFallsBackToAscii);
        registerTest("factory_falls_back_when_missing",
                     testFactoryFallsBackWhenMissing);
        registerTest("bpe_encode_decode_and_special_token",
                     testBpeEncodeDecodeAndSpecialToken);
    }
} tokenizer_test_registrar;