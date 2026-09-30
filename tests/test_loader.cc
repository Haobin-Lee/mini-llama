// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// loader 测试：parseManifest 正常/畸形清单校验 + loadModel 正常与失败路径。

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "tests/test_main.h"

using mini_llama::loadModel;
using mini_llama::ModelManifest;
using mini_llama::parseManifest;

namespace {

// 临时 model.json 文件，析构时清理所在目录
struct TempJson {
    std::filesystem::path dir;
    std::filesystem::path path;

    ~TempJson() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
};

TempJson writeTempJson(const std::string& content) {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    TempJson tmp;
    tmp.dir = std::filesystem::temp_directory_path() /
              ("mini_llama_loader_" + std::to_string(stamp));
    std::filesystem::create_directories(tmp.dir);
    tmp.path = tmp.dir / "model.json";

    std::ofstream out(tmp.path);
    out << content;
    return tmp;
}

// 最小合法 config（n_layers=1，dim=4，head_dim=2）+ ascii tokenizer
const char* kMinimalHead = R"({
  "config": {"vocab_size": 8, "dim": 4, "hidden_dim": 8, "n_layers": 1,
             "n_heads": 2, "n_kv_heads": 2, "max_seq_len": 16,
             "rope_theta": 10000.0, "rms_norm_eps": 1e-05},
  "tokenizer": {"type": "ascii", "bos_id": 1, "eos_id": 2, "unk_id": 0},
)";

}  // namespace

// ---------------------------------------------------------------------------
// parseManifest：合法清单
// ---------------------------------------------------------------------------
static bool testParseManifestTiny() {
    ModelManifest m = parseManifest("models/tiny/model.json");
    MINI_LLAMA_ASSERT_TRUE(m.ok);

    MINI_LLAMA_ASSERT_EQ(m.config.vocab_size, 128);
    MINI_LLAMA_ASSERT_EQ(m.config.dim, 32);
    MINI_LLAMA_ASSERT_EQ(m.config.hidden_dim, 86);
    MINI_LLAMA_ASSERT_EQ(m.config.n_layers, 2);
    MINI_LLAMA_ASSERT_EQ(m.config.n_heads, 4);
    MINI_LLAMA_ASSERT_EQ(m.config.n_kv_heads, 4);
    MINI_LLAMA_ASSERT_EQ(m.config.head_dim, 8);
    MINI_LLAMA_ASSERT_EQ(m.config.max_seq_len, 128);

    MINI_LLAMA_ASSERT_EQ(m.tokenizer.type, std::string("json_vocab"));
    MINI_LLAMA_ASSERT_EQ(m.tokenizer.bos_id, 1);
    MINI_LLAMA_ASSERT_EQ(m.tokenizer.eos_id, 2);
    MINI_LLAMA_ASSERT_EQ(m.tokenizer.unk_id, 0);

    // 3 个全局张量 + 2 层 × 9 个层内张量
    MINI_LLAMA_ASSERT_EQ(static_cast<int>(m.tensors.size()), 21);

    bool found_embedding = false;
    for (const auto& info : m.tensors) {
        if (info.name == "token_embedding") {
            found_embedding = true;
            MINI_LLAMA_ASSERT_TRUE(info.shape == std::vector<int>({128, 32}));
            MINI_LLAMA_ASSERT_EQ(info.dtype, std::string("float32"));
            MINI_LLAMA_ASSERT_EQ(info.offset, static_cast<size_t>(0));
            MINI_LLAMA_ASSERT_EQ(info.byte_size, static_cast<size_t>(16384));
        }
    }
    MINI_LLAMA_ASSERT_TRUE(found_embedding);
    return true;
}

static bool testParseManifestMissingFile() {
    ModelManifest m = parseManifest("models/tiny/does_not_exist.json");
    MINI_LLAMA_ASSERT_TRUE(!m.ok);
    return true;
}

// ---------------------------------------------------------------------------
// parseManifest：畸形清单（全部应 ok=false）
// ---------------------------------------------------------------------------
static bool testParseManifestTruncatedJson() {
    TempJson tmp = writeTempJson("{\"config\": {");
    ModelManifest m = parseManifest(tmp.path.string());
    MINI_LLAMA_ASSERT_TRUE(!m.ok);
    return true;
}

static bool testParseManifestMissingTensorsField() {
    TempJson tmp =
        writeTempJson(std::string(kMinimalHead) + "  \"nothing\": 1\n}");
    ModelManifest m = parseManifest(tmp.path.string());
    MINI_LLAMA_ASSERT_TRUE(!m.ok);
    return true;
}

static bool testParseManifestEmptyTensors() {
    TempJson tmp =
        writeTempJson(std::string(kMinimalHead) + "  \"tensors\": []\n}");
    ModelManifest m = parseManifest(tmp.path.string());
    MINI_LLAMA_ASSERT_TRUE(!m.ok);
    return true;
}

static bool testParseManifestMissingRequiredTensor() {
    // dummy 张量本身合法，但缺少 token_embedding 等必需张量
    TempJson tmp = writeTempJson(
        std::string(kMinimalHead) +
        "  \"tensors\": [{\"name\": \"dummy\", \"shape\": [2], "
        "\"dtype\": \"float32\", \"offset\": 0, \"byte_size\": 8}]\n}");
    ModelManifest m = parseManifest(tmp.path.string());
    MINI_LLAMA_ASSERT_TRUE(!m.ok);
    return true;
}

static bool testParseManifestByteSizeMismatch() {
    TempJson tmp = writeTempJson(
        std::string(kMinimalHead) +
        "  \"tensors\": [{\"name\": \"dummy\", \"shape\": [2], "
        "\"dtype\": \"float32\", \"offset\": 0, \"byte_size\": 4}]\n}");
    ModelManifest m = parseManifest(tmp.path.string());
    MINI_LLAMA_ASSERT_TRUE(!m.ok);
    return true;
}

static bool testParseManifestDuplicateTensor() {
    TempJson tmp = writeTempJson(
        std::string(kMinimalHead) +
        "  \"tensors\": [{\"name\": \"dummy\", \"shape\": [2], "
        "\"dtype\": \"float32\", \"offset\": 0, \"byte_size\": 8},"
        "{\"name\": \"dummy\", \"shape\": [2], "
        "\"dtype\": \"float32\", \"offset\": 8, \"byte_size\": 8}]\n}");
    ModelManifest m = parseManifest(tmp.path.string());
    MINI_LLAMA_ASSERT_TRUE(!m.ok);
    return true;
}

static bool testParseManifestOverlappingTensors() {
    TempJson tmp = writeTempJson(
        std::string(kMinimalHead) +
        "  \"tensors\": [{\"name\": \"dummy_a\", \"shape\": [4], "
        "\"dtype\": \"float32\", \"offset\": 0, \"byte_size\": 16},"
        "{\"name\": \"dummy_b\", \"shape\": [4], "
        "\"dtype\": \"float32\", \"offset\": 8, \"byte_size\": 16}]\n}");
    ModelManifest m = parseManifest(tmp.path.string());
    MINI_LLAMA_ASSERT_TRUE(!m.ok);
    return true;
}

// ---------------------------------------------------------------------------
// loadModel
// ---------------------------------------------------------------------------
static bool testLoadModelTiny() {
    auto model = loadModel("models/tiny/model.json", "models/tiny/model.bin");
    MINI_LLAMA_ASSERT_TRUE(model.loaded);
    MINI_LLAMA_ASSERT_TRUE(model.load_error.empty());
    MINI_LLAMA_ASSERT_EQ(model.layers.size(), static_cast<size_t>(2));

    MINI_LLAMA_ASSERT_TRUE(model.token_embedding.shape ==
                           std::vector<int>({128, 32}));
    MINI_LLAMA_ASSERT_TRUE(model.final_norm.shape == std::vector<int>({32}));
    MINI_LLAMA_ASSERT_TRUE(model.lm_head.shape == std::vector<int>({128, 32}));
    MINI_LLAMA_ASSERT_TRUE(model.layers[0].attention_norm.shape ==
                           std::vector<int>({32}));
    return true;
}

static bool testLoadModelMissingWeights() {
    auto model =
        loadModel("models/tiny/model.json", "models/tiny/does_not_exist.bin");
    MINI_LLAMA_ASSERT_TRUE(!model.loaded);
    MINI_LLAMA_ASSERT_TRUE(!model.load_error.empty());
    return true;
}

static bool testLoadModelMissingConfig() {
    auto model =
        loadModel("models/tiny/does_not_exist.json", "models/tiny/model.bin");
    MINI_LLAMA_ASSERT_TRUE(!model.loaded);
    MINI_LLAMA_ASSERT_TRUE(!model.load_error.empty());
    return true;
}

// ---------------------------------------------------------------------------
// 注册
// ---------------------------------------------------------------------------
static struct LoaderTestRegistrar {
    LoaderTestRegistrar() {
        registerTest("parse_manifest_tiny", testParseManifestTiny);
        registerTest("parse_manifest_missing_file",
                     testParseManifestMissingFile);
        registerTest("parse_manifest_truncated_json",
                     testParseManifestTruncatedJson);
        registerTest("parse_manifest_missing_tensors_field",
                     testParseManifestMissingTensorsField);
        registerTest("parse_manifest_empty_tensors",
                     testParseManifestEmptyTensors);
        registerTest("parse_manifest_missing_required_tensor",
                     testParseManifestMissingRequiredTensor);
        registerTest("parse_manifest_byte_size_mismatch",
                     testParseManifestByteSizeMismatch);
        registerTest("parse_manifest_duplicate_tensor",
                     testParseManifestDuplicateTensor);
        registerTest("parse_manifest_overlapping_tensors",
                     testParseManifestOverlappingTensors);
        registerTest("load_model_tiny", testLoadModelTiny);
        registerTest("load_model_missing_weights", testLoadModelMissingWeights);
        registerTest("load_model_missing_config", testLoadModelMissingConfig);
    }
} loader_test_registrar;
