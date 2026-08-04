// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#include "mini_llama/loader.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "errlog/bizlog.h"

// Simple JSON parser helper for model config and tensor metadata
// Only supports the specific format we need

namespace mini_llama {

using errlog::ErrorCode;

static MiniLlamaModel loadFailure(const std::string& message) {
    MiniLlamaModel model;
    model.loaded = false;
    model.load_error = message;
    std::cerr << "Failed to load model: " << message << std::endl;
    return model;
}

static size_t findConfigKey(const std::string& content, const std::string& key,
                            bool& ok) {
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if (key_pos == std::string::npos) {
        BIZLOG(ErrorCode::kLoaderParseError, "missing config key: " + key);
        ok = false;
        return std::string::npos;
    }
    size_t colon_pos = content.find(":", key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "missing ':' after config key: " + key);
        ok = false;
        return std::string::npos;
    }
    size_t value_pos = colon_pos + 1;
    while (value_pos < content.size() &&
           std::isspace(static_cast<unsigned char>(content[value_pos]))) {
        ++value_pos;
    }
    if (value_pos >= content.size()) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "missing value for config key: " + key);
        ok = false;
        return std::string::npos;
    }
    return value_pos;
}

static bool isJsonValueTerminator(char c) {
    return std::isspace(static_cast<unsigned char>(c)) || c == ',' ||
           c == '}' || c == ']';
}

static void validateNumberTerminator(const std::string& content,
                                     size_t value_pos, size_t parsed,
                                     const std::string& key, bool& ok) {
    size_t pos = value_pos + parsed;
    while (pos < content.size() &&
           std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    if (pos < content.size() && !isJsonValueTerminator(content[pos])) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid numeric value for config key: " + key);
        ok = false;
    }
}

static int parseRequiredInt(const std::string& content, const std::string& key,
                            bool& ok) {
    size_t value_pos = findConfigKey(content, key, ok);
    if (!ok) {
        return 0;
    }
    size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid integer for config key: " + key);
        ok = false;
        return 0;
    }
    if (parsed == 0) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid integer for config key: " + key);
        ok = false;
        return 0;
    }
    validateNumberTerminator(content, value_pos, parsed, key, ok);
    return value;
}

static std::string trimWs(const std::string& text) {
    size_t start = text.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = text.find_last_not_of(" \t\n\r");
    return text.substr(start, end - start + 1);
}

static int parseStrictIntToken(const std::string& token,
                               const std::string& field, bool& ok) {
    std::string trimmed = trimWs(token);
    if (trimmed.empty()) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "missing integer value for " + field);
        ok = false;
        return 0;
    }

    char* end = nullptr;
    errno = 0;
    int64_t value = std::strtoll(trimmed.c_str(), &end, 10);
    if (end == trimmed.c_str() || errno == ERANGE ||
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid integer value for " + field + ": " + trimmed);
        ok = false;
        return 0;
    }
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }
    if (*end != '\0') {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid integer value for " + field + ": " + trimmed);
        ok = false;
        return 0;
    }
    return static_cast<int>(value);
}

static size_t parseStrictSizeValue(const std::string& content, size_t value_pos,
                                   const std::string& field, bool& ok) {
    while (value_pos < content.size() &&
           std::isspace(static_cast<unsigned char>(content[value_pos]))) {
        ++value_pos;
    }
    if (value_pos >= content.size() || content[value_pos] == '-') {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid non-negative integer for " + field);
        ok = false;
        return 0;
    }

    size_t parsed = 0;
    uint64_t value = 0;
    try {
        value = std::stoull(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid non-negative integer for " + field);
        ok = false;
        return 0;
    }
    if (parsed == 0 || value > std::numeric_limits<size_t>::max()) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid non-negative integer for " + field);
        ok = false;
        return 0;
    }
    validateNumberTerminator(content, value_pos, parsed, field, ok);
    return static_cast<size_t>(value);
}

static float parseRequiredFloat(const std::string& content,
                                const std::string& key, bool& ok) {
    size_t value_pos = findConfigKey(content, key, ok);
    if (!ok) {
        return 0.0f;
    }
    size_t parsed = 0;
    float value = 0.0f;
    try {
        value = std::stof(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid float for config key: " + key);
        ok = false;
        return 0.0f;
    }
    if (parsed == 0) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid float for config key: " + key);
        ok = false;
        return 0.0f;
    }
    validateNumberTerminator(content, value_pos, parsed, key, ok);
    return value;
}

static size_t findOptionalKey(const std::string& content,
                              const std::string& key, bool& ok) {
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if (key_pos == std::string::npos) {
        return std::string::npos;
    }
    size_t colon_pos = content.find(":", key_pos + marker.size());
    if (colon_pos == std::string::npos) {
        BIZLOG(ErrorCode::kLoaderParseError, "missing ':' after key: " + key);
        ok = false;
        return std::string::npos;
    }
    size_t value_pos = colon_pos + 1;
    while (value_pos < content.size() &&
           std::isspace(static_cast<unsigned char>(content[value_pos]))) {
        ++value_pos;
    }
    if (value_pos >= content.size()) {
        BIZLOG(ErrorCode::kLoaderParseError, "missing value for key: " + key);
        ok = false;
        return std::string::npos;
    }
    return value_pos;
}

static std::string parseOptionalStringField(const std::string& content,
                                            const std::string& key,
                                            const std::string& fallback,
                                            bool& ok) {
    size_t value_pos = findOptionalKey(content, key, ok);
    if (!ok || value_pos == std::string::npos) {
        return fallback;
    }
    if (content[value_pos] != '"') {
        BIZLOG(ErrorCode::kLoaderParseError,
               "expected string value for key: " + key);
        ok = false;
        return fallback;
    }
    size_t end = content.find('"', value_pos + 1);
    if (end == std::string::npos) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "unterminated string value for key: " + key);
        ok = false;
        return fallback;
    }
    return content.substr(value_pos + 1, end - value_pos - 1);
}

static int parseOptionalIntField(const std::string& content,
                                 const std::string& key, int fallback,
                                 bool& ok) {
    size_t value_pos = findOptionalKey(content, key, ok);
    if (!ok || value_pos == std::string::npos) {
        return fallback;
    }
    size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid integer for key: " + key);
        ok = false;
        return fallback;
    }
    if (parsed == 0) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "invalid integer for key: " + key);
        ok = false;
        return fallback;
    }
    validateNumberTerminator(content, value_pos, parsed, key, ok);
    return value;
}

static std::string parseObjectForKey(const std::string& content,
                                     const std::string& key, bool& ok) {
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if (key_pos == std::string::npos) {
        return "";
    }
    size_t obj_start = content.find("{", key_pos + marker.size());
    if (obj_start == std::string::npos) {
        BIZLOG(ErrorCode::kLoaderParseError, "missing object for key: " + key);
        ok = false;
        return "";
    }

    int depth = 0;
    for (size_t pos = obj_start; pos < content.size(); ++pos) {
        if (content[pos] == '{') {
            ++depth;
        } else if (content[pos] == '}') {
            --depth;
            if (depth == 0) {
                return content.substr(obj_start, pos - obj_start + 1);
            }
        }
    }
    BIZLOG(ErrorCode::kLoaderParseError, "unclosed object for key: " + key);
    ok = false;
    return "";
}

static TokenizerInfo parseTokenizerJson(const std::string& content, bool& ok) {
    TokenizerInfo tokenizer;
    std::string obj = parseObjectForKey(content, "tokenizer", ok);
    if (!ok) {
        return tokenizer;
    }
    if (obj.empty()) {
        return tokenizer;
    }

    tokenizer.type =
        parseOptionalStringField(obj, "type", tokenizer.type, ok);
    if (!ok) {
        return tokenizer;
    }
    tokenizer.path =
        parseOptionalStringField(obj, "path", tokenizer.path, ok);
    if (!ok) {
        return tokenizer;
    }
    tokenizer.bos_id =
        parseOptionalIntField(obj, "bos_id", tokenizer.bos_id, ok);
    if (!ok) {
        return tokenizer;
    }
    tokenizer.eos_id =
        parseOptionalIntField(obj, "eos_id", tokenizer.eos_id, ok);
    if (!ok) {
        return tokenizer;
    }
    tokenizer.unk_id =
        parseOptionalIntField(obj, "unk_id", tokenizer.unk_id, ok);
    if (!ok) {
        return tokenizer;
    }

    if (tokenizer.type != "ascii" && tokenizer.type != "json_vocab") {
        BIZLOG(ErrorCode::kLoaderParseError,
               "unsupported tokenizer type: " + tokenizer.type);
        ok = false;
        return tokenizer;
    }
    if (tokenizer.bos_id < 0 || tokenizer.eos_id < 0 || tokenizer.unk_id < 0) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "tokenizer special ids must be non-negative");
        ok = false;
        return tokenizer;
    }
    if (tokenizer.bos_id == tokenizer.eos_id ||
        tokenizer.bos_id == tokenizer.unk_id ||
        tokenizer.eos_id == tokenizer.unk_id) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "tokenizer special ids must be distinct");
        ok = false;
        return tokenizer;
    }
    if (tokenizer.type == "json_vocab" && tokenizer.path.empty()) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "json_vocab tokenizer requires a path");
        ok = false;
        return tokenizer;
    }

    return tokenizer;
}

static void validateConfig(const ModelConfig& config, bool& ok) {
    if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
        config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
        config.max_seq_len <= 0) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "all model dimensions must be positive");
        ok = false;
        return;
    }
    if (config.dim % config.n_heads != 0) {
        BIZLOG(ErrorCode::kLoaderParseError, "dim must be divisible by n_heads");
        ok = false;
        return;
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "n_heads must be divisible by n_kv_heads");
        ok = false;
        return;
    }
    if (config.head_dim <= 0 || config.head_dim % 2 != 0) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "head_dim must be a positive even number");
        ok = false;
        return;
    }
    if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
        BIZLOG(ErrorCode::kLoaderParseError, "rope_theta must be positive");
        ok = false;
        return;
    }
    if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0f) {
        BIZLOG(ErrorCode::kLoaderParseError, "rms_norm_eps must be positive");
        ok = false;
        return;
    }
}

static size_t checkedTensorByteSize(const TensorInfo& info, bool& ok) {
    size_t num_elements = 1;
    for (int dim : info.shape) {
        if (dim <= 0) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "invalid tensor shape for " + info.name);
            ok = false;
            return 0;
        }
        const size_t d = static_cast<size_t>(dim);
        if (num_elements > std::numeric_limits<size_t>::max() / d) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "tensor size overflow for " + info.name);
            ok = false;
            return 0;
        }
        num_elements *= d;
    }
    if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "tensor byte size overflow for " + info.name);
        ok = false;
        return 0;
    }
    return num_elements * sizeof(float);
}

static std::vector<std::pair<std::string, std::vector<int>>>
ExpectedTensorShapes(const ModelConfig& config) {
    std::vector<std::pair<std::string, std::vector<int>>> expected = {
        {"token_embedding", {config.vocab_size, config.dim}},
        {"final_norm", {config.dim}},
        {"lm_head", {config.vocab_size, config.dim}},
    };

    for (int layer = 0; layer < config.n_layers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        expected.push_back({prefix + "attention_norm", {config.dim}});
        expected.push_back(
            {prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
        expected.push_back(
            {prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
        expected.push_back(
            {prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});
        expected.push_back(
            {prefix + "wo", {config.dim, config.n_heads * config.head_dim}});
        expected.push_back({prefix + "ffn_norm", {config.dim}});
        expected.push_back(
            {prefix + "w_gate", {config.hidden_dim, config.dim}});
        expected.push_back({prefix + "w_up", {config.hidden_dim, config.dim}});
        expected.push_back(
            {prefix + "w_down", {config.dim, config.hidden_dim}});
    }

    return expected;
}

static std::string shapeToString(const std::vector<int>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

static ModelConfig parseConfigJson(const std::string& content, bool& ok) {
    ModelConfig config;
    config.vocab_size = parseRequiredInt(content, "vocab_size", ok);
    if (!ok) {
        return config;
    }
    config.dim = parseRequiredInt(content, "dim", ok);
    if (!ok) {
        return config;
    }
    config.hidden_dim = parseRequiredInt(content, "hidden_dim", ok);
    if (!ok) {
        return config;
    }
    config.n_layers = parseRequiredInt(content, "n_layers", ok);
    if (!ok) {
        return config;
    }
    config.n_heads = parseRequiredInt(content, "n_heads", ok);
    if (!ok) {
        return config;
    }
    config.n_kv_heads = parseRequiredInt(content, "n_kv_heads", ok);
    if (!ok) {
        return config;
    }
    config.max_seq_len = parseRequiredInt(content, "max_seq_len", ok);
    if (!ok) {
        return config;
    }
    config.rope_theta = parseRequiredFloat(content, "rope_theta", ok);
    if (!ok) {
        return config;
    }
    config.rms_norm_eps = parseRequiredFloat(content, "rms_norm_eps", ok);
    if (!ok) {
        return config;
    }
    config.head_dim = config.dim / config.n_heads;

    validateConfig(config, ok);
    return config;
}

// ---------------------------------------------------------------------------
// Parse tensor metadata array from JSON content
// ---------------------------------------------------------------------------
static std::vector<TensorInfo> parseTensorMetadata(const std::string& content,
                                                   bool& ok) {
    std::vector<TensorInfo> result;

    const std::string tensors_marker = "\"tensors\"";
    size_t tensors_pos = content.find(tensors_marker);
    if (tensors_pos == std::string::npos) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "missing 'tensors' field in model.json");
        ok = false;
        return result;
    }

    size_t bracket_open = content.find("[", tensors_pos);
    if (bracket_open == std::string::npos) {
        BIZLOG(ErrorCode::kLoaderParseError, "missing '[' after 'tensors'");
        ok = false;
        return result;
    }

    size_t i = bracket_open + 1;
    int depth = 1;
    while (i < content.size() && depth > 0) {
        // Find next object start
        while (i < content.size() && content[i] != '{') {
            if (content[i] == '[') {
                ++depth;
            } else if (content[i] == ']') {
                --depth;
            }
            ++i;
        }
        if (depth <= 0 || i >= content.size()) {
            break;
        }

        // Parse one tensor object
        TensorInfo info;
        size_t obj_start = i;
        size_t obj_end = content.find("}", obj_start);
        if (obj_end == std::string::npos) {
            BIZLOG(ErrorCode::kLoaderParseError, "unclosed tensor object");
            ok = false;
            return result;
        }
        std::string obj = content.substr(obj_start, obj_end - obj_start + 1);

        bool has_name = false;
        bool has_dtype = false;
        bool has_shape = false;
        bool has_offset = false;
        bool has_byte_size = false;

        // Parse name
        size_t name_pos = obj.find("\"name\"");
        if (name_pos != std::string::npos) {
            size_t name_colon = obj.find(":", name_pos);
            size_t name_quote = obj.find("\"", name_colon + 1);
            size_t name_end = obj.find("\"", name_quote + 1);
            if (name_quote != std::string::npos &&
                name_end != std::string::npos) {
                info.name =
                    obj.substr(name_quote + 1, name_end - name_quote - 1);
                has_name = true;
            }
        }

        // Parse dtype
        size_t dtype_pos = obj.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t dtype_colon = obj.find(":", dtype_pos);
            size_t dtype_quote = obj.find("\"", dtype_colon + 1);
            size_t dtype_end = obj.find("\"", dtype_quote + 1);
            if (dtype_quote != std::string::npos &&
                dtype_end != std::string::npos) {
                info.dtype =
                    obj.substr(dtype_quote + 1, dtype_end - dtype_quote - 1);
                has_dtype = true;
            }
        }

        // Parse shape array
        size_t shape_pos = obj.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t shape_bracket = obj.find("[", shape_pos);
            size_t shape_end = obj.find("]", shape_bracket);
            if (shape_bracket != std::string::npos &&
                shape_end != std::string::npos) {
                std::string shape_string_short = obj.substr(
                    shape_bracket + 1, shape_end - shape_bracket - 1);
                std::stringstream ss(shape_string_short);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    // Trim whitespace
                    size_t start = token.find_first_not_of(" \t\n\r");
                    size_t end = token.find_last_not_of(" \t\n\r");
                    if (start != std::string::npos) {
                        int dim =
                            std::stoi(token.substr(start, end - start + 1));
                        info.shape.push_back(dim);
                    }
                }
                has_shape = true;
            }
        }

        // Parse offset
        size_t offset_pos = obj.find("\"offset\"");
        if (offset_pos != std::string::npos) {
            size_t offset_colon = obj.find(":", offset_pos);
            if (offset_colon == std::string::npos) {
                BIZLOG(ErrorCode::kLoaderParseError,
                       "tensor offset field is malformed");
                ok = false;
                return result;
            }
            info.offset =
                static_cast<size_t>(std::stoull(obj.substr(offset_colon + 1)));
            has_offset = true;
        }

        // Parse byte_size
        size_t bs_pos = obj.find("\"byte_size\"");
        if (bs_pos != std::string::npos) {
            size_t bs_colon = obj.find(":", bs_pos);
            if (bs_colon == std::string::npos) {
                BIZLOG(ErrorCode::kLoaderParseError,
                       "tensor byte_size field is malformed");
                ok = false;
                return result;
            }
            info.byte_size =
                static_cast<size_t>(std::stoull(obj.substr(bs_colon + 1)));
            has_byte_size = true;
        }

        if (!has_name || info.name.empty()) {
            BIZLOG(ErrorCode::kLoaderParseError, "tensor missing name field");
            ok = false;
            return result;
        }
        if (!has_dtype) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "tensor missing dtype field: " + info.name);
            ok = false;
            return result;
        }
        if (!has_shape) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "tensor missing shape field: " + info.name);
            ok = false;
            return result;
        }
        if (!has_offset) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "tensor missing offset field: " + info.name);
            ok = false;
            return result;
        }
        if (!has_byte_size) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "tensor missing byte_size field: " + info.name);
            ok = false;
            return result;
        }

        result.push_back(info);
        i = obj_end + 1;
    }

    return result;
}

static void validateManifest(const ModelManifest& manifest, bool& ok) {
    if (manifest.tensors.empty()) {
        BIZLOG(ErrorCode::kLoaderParseError, "model.json contains no tensors");
        ok = false;
        return;
    }

    std::map<std::string, TensorInfo> tensor_map;
    std::vector<TensorInfo> by_offset;
    by_offset.reserve(manifest.tensors.size());

    for (const auto& info : manifest.tensors) {
        if (info.dtype != "float32") {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "unsupported dtype '" + info.dtype +
                       "' for tensor: " + info.name);
            ok = false;
            return;
        }
        if (info.shape.empty()) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "tensor shape must not be empty: " + info.name);
            ok = false;
            return;
        }
        size_t expected_bytes = checkedTensorByteSize(info, ok);
        if (!ok) {
            return;
        }
        if (expected_bytes != info.byte_size) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "tensor byte_size mismatch for " + info.name + ": expected " +
                       std::to_string(expected_bytes) + ", manifest says " +
                       std::to_string(info.byte_size));
            ok = false;
            return;
        }
        if (info.offset > std::numeric_limits<size_t>::max() - info.byte_size) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "tensor offset overflow for " + info.name);
            ok = false;
            return;
        }
        auto inserted = tensor_map.emplace(info.name, info);
        if (!inserted.second) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "duplicate tensor name: " + info.name);
            ok = false;
            return;
        }
        by_offset.push_back(info);
    }

    std::sort(by_offset.begin(), by_offset.end(),
              [](const TensorInfo& a, const TensorInfo& b) {
                  return a.offset < b.offset;
              });
    for (size_t i = 1; i < by_offset.size(); ++i) {
        size_t previous_end =
            by_offset[i - 1].offset + by_offset[i - 1].byte_size;
        if (by_offset[i].offset < previous_end) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "tensor ranges overlap: " + by_offset[i - 1].name + " and " +
                       by_offset[i].name);
            ok = false;
            return;
        }
    }

    for (const auto& expected : ExpectedTensorShapes(manifest.config)) {
        auto it = tensor_map.find(expected.first);
        if (it == tensor_map.end()) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "missing required tensor: " + expected.first);
            ok = false;
            return;
        }
        if (it->second.shape != expected.second) {
            BIZLOG(ErrorCode::kLoaderParseError,
                   "tensor shape mismatch for " + expected.first +
                       ": expected " + shapeToString(expected.second) +
                       ", manifest says " + shapeToString(it->second.shape));
            ok = false;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// ModelManifest
// ---------------------------------------------------------------------------
ModelManifest parseManifest(const std::string& config_path) {
    ModelManifest manifest;

    std::ifstream cf(config_path);
    if (!cf.is_open()) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "failed to open config: " + config_path);
        manifest.ok = false;
        return manifest;
    }
    std::stringstream buffer;
    buffer << cf.rdbuf();
    if (cf.bad()) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "failed to read config: " + config_path);
        manifest.ok = false;
        return manifest;
    }

    std::string content = buffer.str();
    bool ok = true;

    manifest.config = parseConfigJson(content, ok);
    if (!ok) {
        manifest.ok = false;
        return manifest;
    }
    manifest.tokenizer = parseTokenizerJson(content, ok);
    if (!ok) {
        manifest.ok = false;
        return manifest;
    }
    manifest.tensors = parseTensorMetadata(content, ok);
    if (!ok) {
        manifest.ok = false;
        return manifest;
    }
    validateManifest(manifest, ok);
    if (!ok) {
        manifest.ok = false;
        return manifest;
    }

    return manifest;
}

// ---------------------------------------------------------------------------
// Load a single tensor from binary file using manifest metadata
// ---------------------------------------------------------------------------
static Tensor loadTensorByName(
    std::ifstream& wf, const std::map<std::string, TensorInfo>& tensor_map,
    const std::string& name, bool& ok) {
    auto it = tensor_map.find(name);
    if (it == tensor_map.end()) {
        BIZLOG(ErrorCode::kLoaderParseError, "missing required tensor: " + name);
        ok = false;
        return Tensor({1}, 0.0f);
    }

    const TensorInfo& info = it->second;

    if (info.offset >
        static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "tensor offset exceeds stream limit for " + name);
        ok = false;
        return Tensor({1}, 0.0f);
    }
    if (info.byte_size >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "tensor byte size exceeds stream limit for " + name);
        ok = false;
        return Tensor({1}, 0.0f);
    }

    wf.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
    if (!wf.good()) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "failed to seek to offset for tensor: " + name);
        ok = false;
        return Tensor({1}, 0.0f);
    }

    Tensor t(info.shape, 0.0f);
    wf.read(reinterpret_cast<char*>(t.data.data()),
            static_cast<std::streamsize>(info.byte_size));
    if (static_cast<size_t>(wf.gcount()) != info.byte_size) {
        BIZLOG(ErrorCode::kLoaderParseError,
               "weights file ended while reading tensor: " + name);
        ok = false;
        return Tensor({1}, 0.0f);
    }

    return t;
}

// ---------------------------------------------------------------------------
// LoadModel
// ---------------------------------------------------------------------------
MiniLlamaModel loadModel(const std::string& config_path,
                         const std::string& weights_path) {
    ModelManifest manifest = parseManifest(config_path);
    if (!manifest.ok) {
        return loadFailure("failed to parse manifest: " + config_path);
    }

    std::cout << "Loaded config: vocab=" << manifest.config.vocab_size
              << " dim=" << manifest.config.dim
              << " layers=" << manifest.config.n_layers
              << " heads=" << manifest.config.n_heads
              << " kv_heads=" << manifest.config.n_kv_heads
              << " head_dim=" << manifest.config.head_dim
              << " hidden_dim=" << manifest.config.hidden_dim << std::endl;

    std::map<std::string, TensorInfo> tensor_map;
    for (const auto& info : manifest.tensors) {
        tensor_map[info.name] = info;
    }

    // Open weights binary
    std::ifstream wf(weights_path, std::ios::binary);
    if (!wf.is_open()) {
        return loadFailure("failed to open weights: " + weights_path);
    }

    MiniLlamaModel model;
    model.config = manifest.config;

    bool ok = true;
    model.token_embedding =
        loadTensorByName(wf, tensor_map, "token_embedding", ok);
    if (!ok) {
        return loadFailure("failed to load tensor: token_embedding");
    }

    model.layers.resize(manifest.config.n_layers);
    for (int layer = 0; layer < manifest.config.n_layers; ++layer) {
        LayerWeights& lw = model.layers[layer];
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        lw.attention_norm =
            loadTensorByName(wf, tensor_map, prefix + "attention_norm", ok);
        if (!ok) {
            return loadFailure("failed to load tensor: " + prefix +
                               "attention_norm");
        }
        lw.wq = toQuantizedTensor(
            loadTensorByName(wf, tensor_map, prefix + "wq", ok));
        if (!ok) {
            return loadFailure("failed to load tensor: " + prefix + "wq");
        }
        lw.wk = toQuantizedTensor(
            loadTensorByName(wf, tensor_map, prefix + "wk", ok));
        if (!ok) {
            return loadFailure("failed to load tensor: " + prefix + "wk");
        }
        lw.wv = toQuantizedTensor(
            loadTensorByName(wf, tensor_map, prefix + "wv", ok));
        if (!ok) {
            return loadFailure("failed to load tensor: " + prefix + "wv");
        }
        lw.wo = toQuantizedTensor(
            loadTensorByName(wf, tensor_map, prefix + "wo", ok));
        if (!ok) {
            return loadFailure("failed to load tensor: " + prefix + "wo");
        }
        lw.ffn_norm =
            loadTensorByName(wf, tensor_map, prefix + "ffn_norm", ok);
        if (!ok) {
            return loadFailure("failed to load tensor: " + prefix + "ffn_norm");
        }
        lw.w_gate = toQuantizedTensor(
            loadTensorByName(wf, tensor_map, prefix + "w_gate", ok));
        if (!ok) {
            return loadFailure("failed to load tensor: " + prefix + "w_gate");
        }
        lw.w_up = toQuantizedTensor(
            loadTensorByName(wf, tensor_map, prefix + "w_up", ok));
        if (!ok) {
            return loadFailure("failed to load tensor: " + prefix + "w_up");
        }
        lw.w_down = toQuantizedTensor(
            loadTensorByName(wf, tensor_map, prefix + "w_down", ok));
        if (!ok) {
            return loadFailure("failed to load tensor: " + prefix + "w_down");
        }
    }

    model.final_norm = loadTensorByName(wf, tensor_map, "final_norm", ok);
    if (!ok) {
        return loadFailure("failed to load tensor: final_norm");
    }
    model.lm_head =
        toQuantizedTensor(loadTensorByName(wf, tensor_map, "lm_head", ok));
    if (!ok) {
        return loadFailure("failed to load tensor: lm_head");
    }

    // Validate trailing bytes
    wf.seekg(0, std::ios::end);
    std::streamoff file_size = wf.tellg();
    if (file_size < 0) {
        return loadFailure("failed to determine weights file size");
    }

    size_t expected_end = 0;
    for (const auto& info : manifest.tensors) {
        size_t tensor_end = info.offset + info.byte_size;
        if (tensor_end > expected_end) {
            expected_end = tensor_end;
        }
    }

    if (static_cast<size_t>(file_size) != expected_end) {
        if (static_cast<size_t>(file_size) > expected_end) {
            return loadFailure("weights file has trailing bytes: file size=" +
                               std::to_string(file_size) +
                               ", expected=" + std::to_string(expected_end));
        } else {
            return loadFailure("weights file is too short: file size=" +
                               std::to_string(file_size) +
                               ", expected=" + std::to_string(expected_end));
        }
    }

    model.loaded = true;
    std::cout << "Model loaded successfully from " << weights_path << std::endl;
    return model;
}

// ---------------------------------------------------------------------------
// InspectModel: 打印模型元数据
// ---------------------------------------------------------------------------
bool inspectModel(const std::string& config_path) {
    ModelManifest manifest = parseManifest(config_path);
    if (!manifest.ok) {
        std::cerr << "Failed to inspect model: " << config_path << std::endl;
        return false;
    }

    const ModelConfig& c = manifest.config;
    std::cout << "=== Model Config ===" << std::endl;
    std::cout << "  vocab_size:    " << c.vocab_size << std::endl;
    std::cout << "  dim:           " << c.dim << std::endl;
    std::cout << "  hidden_dim:    " << c.hidden_dim << std::endl;
    std::cout << "  n_layers:      " << c.n_layers << std::endl;
    std::cout << "  n_heads:       " << c.n_heads << std::endl;
    std::cout << "  n_kv_heads:    " << c.n_kv_heads << std::endl;
    std::cout << "  head_dim:      " << c.head_dim << std::endl;
    std::cout << "  max_seq_len:   " << c.max_seq_len << std::endl;
    std::cout << "  rope_theta:    " << c.rope_theta << std::endl;
    std::cout << "  rms_norm_eps:  " << c.rms_norm_eps << std::endl;

    std::cout << std::endl;
    std::cout << "=== Tensors (" << manifest.tensors.size()
              << ") ===" << std::endl;

    size_t total_bytes = 0;
    for (const auto& t : manifest.tensors) {
        std::cout << "  " << t.name << std::endl;
        std::cout << "    shape:     [";
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << t.shape[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "    dtype:     " << t.dtype << std::endl;
        std::cout << "    offset:    " << t.offset << std::endl;
        std::cout << "    byte_size: " << t.byte_size << std::endl;
        total_bytes += t.byte_size;
    }

    std::cout << std::endl;
    std::cout << "=== Tokenizer ===" << std::endl;
    std::cout << "  type: " << manifest.tokenizer.type << std::endl;
    std::cout << "  path: "
              << (manifest.tokenizer.path.empty() ? "<none>"
                                                  : manifest.tokenizer.path)
              << std::endl;
    std::cout << "  bos_id: " << manifest.tokenizer.bos_id << std::endl;
    std::cout << "  eos_id: " << manifest.tokenizer.eos_id << std::endl;
    std::cout << "  unk_id: " << manifest.tokenizer.unk_id << std::endl;
    std::cout << std::endl;
    std::cout << "Total tensor bytes: " << total_bytes << std::endl;
    return true;
}

}  // namespace mini_llama
