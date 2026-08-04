// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_LOADER_H_
#define INCLUDE_MINI_LLAMA_LOADER_H_

#include <string>
#include <vector>

#include "mini_llama/model.h"

namespace mini_llama {

struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    size_t offset = 0;
    size_t byte_size = 0;
};

struct TokenizerInfo {
    std::string type = "ascii";
    std::string path;
    int bos_id = 1;
    int eos_id = 2;
    int unk_id = 0;
};

struct ModelManifest {
    ModelConfig config;
    TokenizerInfo tokenizer;
    std::vector<TensorInfo> tensors;
    bool ok = true;
};

// Parse model.json into a manifest (config + tensor metadata).
// On incomplete/inconsistent manifests, logs via BIZLOG and returns a manifest
// with ok=false.
ModelManifest parseManifest(const std::string& config_path);

// Load weights by looking up manifest tensors by name in the weights file.
MiniLlamaModel loadModel(const std::string& config_path,
                         const std::string& weights_path);

// Print model metadata without loading weights.
bool inspectModel(const std::string& config_path);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_LOADER_H_
