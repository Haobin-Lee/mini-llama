// Copyright (c) 2026
// SPDX-License-Identifier: MIT

// Suggested subcommands (see README): generate / run / inspect / bench.
// loader -> tokenizer -> context -> forward -> sampler.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "errlog/log_guard.h"
#include "mini_llama/backend.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/debug.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/ops.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/request_context.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "mini_llama/threadpool.h"
#include "mini_llama/tokenizer.h"

namespace mini_llama {

struct LoadedModel {
    MiniLlamaModel model;
    std::unique_ptr<ITokenizer> tokenizer;
    std::string chat_template;
};

struct LinearWeightTypeCounts {
    size_t f32 = 0;
    size_t q8_0 = 0;
    size_t q4_0 = 0;
    size_t q4_1 = 0;

    size_t total() const { return f32 + q8_0 + q4_0 + q4_1; }

    size_t quantized() const { return q8_0 + q4_0 + q4_1; }
};

static bool parseIntArg(const char* text, int& value) {
    char* end = nullptr;
    int64_t parsed = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    if (parsed < 0 || parsed > 1000000) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static bool parseUintArg(const char* text, unsigned int& value) {
    if (text[0] == '-' || text[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    uint64_t parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    if (errno == ERANGE || parsed > std::numeric_limits<unsigned int>::max()) {
        return false;
    }
    value = static_cast<unsigned int>(parsed);
    return true;
}

static bool parseFloatArg(const char* text, float& value) {
    char* end = nullptr;
    errno = 0;
    double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0') {
        return false;
    }
    if (errno == ERANGE || !std::isfinite(parsed) || parsed < 0.0 ||
        parsed > 10000.0) {
        return false;
    }
    value = static_cast<float>(parsed);
    return true;
}

static bool parseBackendArg(const char* text, BackendConfig& config) {
    BackendKind kind;
    if (!parseBackendKind(text, kind)) {
        return false;
    }
    config.kind = kind;
    return true;
}

static bool parseDeviceArg(const char* text, BackendConfig& config) {
    int device_id = 0;
    if (!parseIntArg(text, device_id) || device_id < 0) {
        return false;
    }
    config.device_id = device_id;
    config.device_id_set = true;
    return true;
}

static bool isGgufFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    char magic[4];
    return f.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}

static std::string findGgufInDirectory(const std::string& path) {
    std::vector<std::filesystem::path> candidates;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            candidates.push_back(entry.path());
        }
    }
    if (candidates.empty()) {
        return "";
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.front().string();
}

static Tensor runLogitsForTokens(const MiniLlamaModel& model,
                                 const std::vector<int>& tokens) {
    MiniLlamaContext ctx(&model);
    MiniBatch batch = MiniBatch::fromTokens(tokens, 0);
    return forwardBatch(ctx, model, batch);
}

static bool applyQuantOverride(MiniLlamaModel& model,
                               const std::string& quant_type) {
    if (quant_type.empty()) {
        return true;
    }
    if (quant_type == "q8_0") {
        quantizeModelToQ80(model);
        return true;
    }
    if (quant_type == "q4_0") {
        quantizeModelToQ40(model);
        return true;
    }
    BIZLOG(ErrorCode::kQuantError,
           "Unsupported quantization type: " + quant_type);
    return false;
}

static std::string resolveManifestTokenizerPath(
    const std::string& config_path) {
    ModelManifest manifest = parseManifest(config_path);
    if (!manifest.ok) {
        return "";
    }
    if (manifest.tokenizer.type != "json_vocab" ||
        manifest.tokenizer.path.empty()) {
        return "";
    }
    std::filesystem::path tokenizer_path(manifest.tokenizer.path);
    if (tokenizer_path.is_relative()) {
        std::filesystem::path config_file(config_path);
        tokenizer_path = config_file.parent_path() / tokenizer_path;
    }
    return tokenizer_path.string();
}

static std::unique_ptr<ITokenizer> createTokenizerFromVocabHint(
    const std::string& vocab_path) {
    std::filesystem::path vocab(vocab_path);
    std::filesystem::path dir = vocab.parent_path();
    std::filesystem::path merges = dir / "merges.txt";
    std::filesystem::path special = dir / "special_tokens.json";
    if (std::filesystem::exists(merges)) {
        return createBpeTokenizer(vocab.string(), merges.string(),
                                  special.string());
    }
    return createTokenizer(vocab.string());
}

static LoadedModel loadModelAndTokenizer(
    const std::string& path, const std::string& explicit_config_path = "",
    const std::string& explicit_tokenizer_path = "") {
    LoadedModel result;

    std::string model_path = path;
    std::string config_path;
    std::string tokenizer_path;
    if (std::filesystem::is_directory(path)) {
        std::filesystem::path bin_path =
            std::filesystem::path(path) / "model.bin";
        std::filesystem::path json_path =
            std::filesystem::path(path) / "model.json";
        std::filesystem::path gguf_path = findGgufInDirectory(path);
        if ((!std::filesystem::exists(bin_path) ||
             !std::filesystem::exists(json_path)) &&
            !gguf_path.empty()) {
            model_path = gguf_path;
        }
    }

    bool is_gguf = isGgufFile(model_path);

    if (is_gguf) {
        result.model = loadGgufModel(model_path);
        if (!result.model.loaded) {
            return result;
        };
        // 1.
        // 先从--tokenizer路径加载tokenizer，如果没有指定，则直接从GGUF文件中加载tokenizer
        if (!explicit_tokenizer_path.empty()) {
            result.tokenizer =
                createTokenizerFromVocabHint(explicit_tokenizer_path);
        } else {
            result.tokenizer = createGgufTokenizer(model_path);
        }

        // 2. 如果gguf中没有tokenizer metadata，则从vocab.json +
        // merges.txt加载tokenizer
        if (!result.tokenizer) {
            std::filesystem::path gguf_dir =
                std::filesystem::path(model_path).parent_path();
            std::string vocab_path = (gguf_dir / "vocab.json").string();
            std::string merges_path = (gguf_dir / "merges.txt").string();
            std::string special_path =
                (gguf_dir / "special_tokens.json").string();
            if (std::filesystem::exists(vocab_path) &&
                std::filesystem::exists(merges_path)) {
                result.tokenizer =
                    createBpeTokenizer(vocab_path, merges_path, special_path);
            }
        }

        // 3. Load chat template from GGUF metadata (M14)
        result.chat_template = loadChatTemplateFromGguf(model_path);
    } else {  // Directory-based JSON+BIN format
        if (std::filesystem::is_directory(path)) {
            model_path = path + "/model.bin";
            config_path = path + "/model.json";
        } else {
            model_path = path;
            if (!explicit_config_path.empty()) {
                config_path = explicit_config_path;
            } else {
                config_path =
                    std::filesystem::path(path).parent_path().string() +
                    "/model.json";
            }
        }
        result.model = loadModel(config_path, model_path);
        if (!result.model.loaded) {
            return result;
        }
        if (!explicit_tokenizer_path.empty()) {
            tokenizer_path = explicit_tokenizer_path;
        }
        if (tokenizer_path.empty()) {
            std::filesystem::path dir =
                std::filesystem::is_directory(path)
                    ? std::filesystem::path(path)
                    : std::filesystem::path(path).parent_path();
            std::filesystem::path auto_vocab = dir / "vocab.json";
            if (std::filesystem::exists(auto_vocab)) {
                tokenizer_path = auto_vocab.string();
            }
        }
        if (tokenizer_path.empty()) {
            tokenizer_path = resolveManifestTokenizerPath(config_path);
        }
        result.tokenizer = std::make_unique<JsonVocabTokenizer>(tokenizer_path);
    }

    if (!result.tokenizer) {
        result.model.load_error = "Failed to load tokenizer";
        result.model.loaded = false;
        return result;
    }

    return result;
}

static size_t commonPrefixLength(const std::vector<int>& a,
                                 const std::vector<int>& b) {
    size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

static void printRequestTrace(const RequestContext& request,
                              std::ostream& out = std::cout) {
    for (const std::string& line : formatRequestTraceEvents(request)) {
        out << line << "\n";
    }
    out << formatRequestTraceSummary(request) << "\n";
}

static bool prepareCudaWeightsOrPrint(MiniLlamaModel& model,
                                      const BackendConfig& config) {
    if (config.kind != BackendKind::kCuda) {
        return true;
    }
    try {
        uploadModelWeightsToCuda(model, config.device_id);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CUDA weight upload failed: " << e.what() << "\n";
        return false;
    }
}

static void printGenerateUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " generate [options]\n"
        << "Options:\n"
        << "  --model <path|dir>   Path to model weights binary or model "
           "directory (default: models/tiny/model.bin)\n"
        << "  --config <path>      Path to model config JSON (default: "
           "models/tiny/model.json)\n"
        << "  -p, --prompt <str>   Input prompt text (default: \"hello\")\n"
        << "  -n, --n-predict <n>  Number of tokens to generate (default: 16)\n"
        << "  --temperature <T>    Sampling temperature (default: 0.0 = "
           "greedy)\n"
        << "  --top-k <k>          Top-k sampling (default: 0 = disabled)\n"
        << "  --seed <S>           Random seed for reproducible sampling "
           "(default: 0 = random)\n"
        << "  --tokenizer <path>   Path to vocab.json tokenizer file\n"
        << "  --quant q8_0|q4_0    Quantize loaded Linear weights before "
           "generation\n"
        << "  --threads <n>        Number of threads for parallel ops (0 = "
           "auto)\n"
        << "  --backend cpu|cuda   Execution backend (default: cpu; cuda "
           "requires -DMINI_LLAMA_CUDA=ON)\n"
        << "  --device <n>         CUDA device id for --backend cuda (default: "
           "0)\n"
        << "  --dump-logits <dir>  Dump logits for each step to directory\n"
        << "  -h, --help           Show this help\n";
}

static bool dumpLogits(const Tensor& logits, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        BIZLOG(ErrorCode::kOpenFileFailed, path);
        return false;
    }
    out.write(reinterpret_cast<const char*>(logits.data.data()),
              static_cast<std::streamsize>(logits.data.size() * sizeof(float)));
    if (!out.good()) {
        BIZLOG(ErrorCode::kWriteFileFailed, path);
        return false;
    }
    return true;
}

static bool checkDumpDirectory(const std::string& path) {
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
        BIZLOG(ErrorCode::kCreateDumpDirectoryFailed, path);
        return false;
    }
    if (!std::filesystem::is_directory(path)) {
        BIZLOG(ErrorCode::kInvalidPath, path);
        return false;
    }
    return true;
}

static bool dumpGeneratedTokens(const std::vector<int>& generated,
                                const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        BIZLOG(ErrorCode::kOpenFileFailed, path);
        return false;
    }
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) {
            out << " ";
        }
        out << generated[i];
    }
    out << "\n";
    if (!out.good()) {
        BIZLOG(ErrorCode::kWriteFileFailed, path);
        return false;
    }
    return true;
}

static int runGenerate(int argc, char** argv) {
    std::string model_path = "models/tiny/model.bin";
    std::string config_path = "models/tiny/model.json";
    bool config_path_set = false;
    std::string prompt = "hello";
    int n_predict = 16;
    float temperature = 0.0f;
    int top_k = 0;
    unsigned int seed = 0;
    std::string dump_logits_dir;
    std::string tokenizer_path;
    std::string quant_type;
    int n_threads = 0;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
            config_path_set = true;
        } else if ((std::strcmp(argv[i], "-p") == 0 ||
                    std::strcmp(argv[i], "--prompt") == 0) &&
                   i + 1 < argc) {
            prompt = argv[++i];
        } else if ((std::strcmp(argv[i], "-n") == 0 ||
                    std::strcmp(argv[i], "--n-predict") == 0) &&
                   i + 1 < argc) {
            if (!parseIntArg(argv[++i], n_predict)) {
                std::cerr << "Invalid --n-predict value. Expected a "
                             "non-negative integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!parseFloatArg(argv[++i], temperature)) {
                std::cerr << "Invalid --temperature value. Expected a "
                             "non-negative float.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parseIntArg(argv[++i], top_k)) {
                std::cerr << "Invalid --top-k value. Expected a non-negative "
                             "integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parseUintArg(argv[++i], seed)) {
                std::cerr << "Invalid --seed value. Expected a non-negative "
                             "integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
            dump_logits_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            quant_type = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            if (!parseIntArg(argv[++i], n_threads) || n_threads < 0) {
                std::cerr << "Invalid --threads value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            printGenerateUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
            printGenerateUsage(argv[0]);
            return 1;
        }
    }

    if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
        std::cerr << "Invalid --quant value: " << quant_type
                  << ". Supported values: q8_0, q4_0.\n";
        return 1;
    }

    std::cout << "mini-llama.cpp\n";
    std::cout << "==============\n\n";

    if (!dump_logits_dir.empty()) {
        if (!checkDumpDirectory(dump_logits_dir)) {
            return 1;
        }
    }

    // For backward compat, if --config was explicitly set but --model is the
    // default, override model_path to be the config's directory.
    if (config_path_set && model_path == "models/tiny/model.bin") {
        model_path = std::filesystem::path(config_path).parent_path().string() +
                     "/model.bin";
    }

    RequestContext request = startRequest("generate", "cpu", model_path);

    auto stage_start = RequestClock::now();
    LoadedModel lm = loadModelAndTokenizer(
        model_path, config_path_set ? config_path : "", tokenizer_path);
    request.model_load_ms = elapsedMs(stage_start);
    request.recordEvent("model_load", request.model_load_ms, 0, model_path);
    if (!lm.model.loaded) {
        request.setError("Failed to load model: " + lm.model.load_error);
        request.finish();
        printRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        request.setError("Failed to load tokenizer.");
        request.finish();
        printRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }
    if (lm.model.config.vocab_size < lm.tokenizer->vocabSize()) {
        request.setError("Model vocab_size must be at least " +
                         std::to_string(lm.tokenizer->vocabSize()) +
                         " for the tokenizer.");
        request.finish();
        printRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }
    MiniLlamaModel& model = lm.model;

    stage_start = RequestClock::now();
    if (!applyQuantOverride(model, quant_type)) {
        request.finish();
        return 1;
    }
    request.recordEvent("quantize", elapsedMs(stage_start), 0,
                        quant_type.empty() ? "model-native" : quant_type);

    // TODO: cuda weight
    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;
    stage_start = RequestClock::now();
    std::vector<int> tokens = tokenizer->encode(prompt);
    request.tokenize_ms = elapsedMs(stage_start);
    request.prompt_tokens = static_cast<int>(tokens.size());
    request.recordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                        "prompt");
    std::cout << "prompt: " << prompt << "\n";
    std::cout << "tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << tokens[i];
    }
    std::cout << "]\n";
    ThreadPool::setThreadCount(n_threads);
    std::cout << "sampling: temperature=" << temperature << ", top_k=" << top_k
              << ", seed=" << seed << "\n";
    std::cout << "quant: " << (quant_type.empty() ? "model-native" : quant_type)
              << "\n";
    std::cout << "threads: " << ThreadPool::getThreadCount() << "\n\n";
    // TODO: print cuda execution summary

    if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
        request.setError("Prompt is too long for max_seq_len=" +
                         std::to_string(model.config.max_seq_len) + ".");
        request.finish();
        printRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }
    if (tokens.size() + static_cast<size_t>(n_predict) >
        static_cast<size_t>(model.config.max_seq_len)) {
        request.setError("Requested tokens exceed context window.");
        request.finish();
        printRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }

    MiniLlamaContext ctx(&model);
    SamplingParams sampling_params;
    sampling_params.temperature = temperature;
    sampling_params.top_k = top_k;
    sampling_params.seed = seed;
    MiniSampler sampler(sampling_params);

    size_t prompt_len = tokens.size();
    int step = 0;
    try {
        Tensor logits;
        std::cout << "prefill...\n";
        stage_start = RequestClock::now();
        if (!dump_logits_dir.empty()) {
            for (size_t i = 0; i < prompt_len; ++i) {
                MiniBatch prefill_step =
                    MiniBatch::fromTokens({tokens[i]}, static_cast<int>(i));
                logits = forwardBatch(ctx, model, prefill_step);
                ++ctx.n_prefill_tokens;
                dumpLogits(logits, dump_logits_dir + "/logits_step" +
                                       std::to_string(step) + ".bin");
                ++step;
            }
        } else {
            MiniBatch prefill = MiniBatch::fromTokens(tokens, 0);
            logits = forwardBatch(ctx, model, prefill);
            ctx.n_prefill_tokens += static_cast<int>(tokens.size());
        }
        request.prefill_ms = elapsedMs(stage_start);
        request.prefill_tokens = static_cast<int>(prompt_len);
        request.recordEvent("prefill", request.prefill_ms,
                            request.prefill_tokens,
                            dump_logits_dir.empty() ? "batch" : "step_dump");

        if (n_predict > 0) {
            stage_start = RequestClock::now();
            int next_token = sampler.sample(logits, sampling_params);
            request.sample_ms += elapsedMs(stage_start);
            tokens.push_back(next_token);

            std::cout << "decode loop...\n";
            for (int i = 1; i < n_predict; ++i) {
                MiniBatch decode_batch = MiniBatch::single(
                    tokens.back(), static_cast<int>(tokens.size() - 1));
                stage_start = RequestClock::now();
                logits = forwardBatch(ctx, model, decode_batch);
                double decode_ms = elapsedMs(stage_start);
                request.decode_ms += decode_ms;
                request.recordEvent("decode", decode_ms, 1,
                                    "pos=" + std::to_string(tokens.size() - 1));
                ++ctx.n_decode_tokens;
                ++request.decode_tokens;
                if (!dump_logits_dir.empty()) {
                    dumpLogits(logits, dump_logits_dir + "/logits_step" +
                                           std::to_string(step) + ".bin");
                    ++step;
                }
                stage_start = RequestClock::now();
                next_token = sampler.sample(logits, sampling_params);
                request.sample_ms += elapsedMs(stage_start);
                tokens.push_back(next_token);
                if (next_token == tokenizer->eosId()) {
                    break;
                }
            }
        } else {
            std::cout << "decode loop skipped.\n";
        }
    } catch (const std::exception& e) {
        request.setError("Inference failed: " + std::string(e.what()));
        request.finish();
        printRequestTrace(request, std::cerr);
        std::cerr << request.error << "\n";
        return 1;
    }

    if (!dump_logits_dir.empty()) {
        std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());

        if (!dumpGeneratedTokens(generated,
                                 dump_logits_dir + "/generation_tokens.txt")) {
            request.finish();
            return 1;
        }
    }

    std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());
    request.generated_tokens = static_cast<int>(generated.size());
    request.recordEvent("sample", request.sample_ms, request.generated_tokens,
                        "generated_tokens");
    request.finish();
    std::cout << "\ngenerated tokens: [";
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << generated[i];
    }
    std::cout << "]\n";

    std::string generated_text = tokenizer->decode(generated);
    std::cout << "generated text: \"" << generated_text << "\"\n";
    // TODO: print cuda summary
    printRequestTrace(request);

    return 0;
}

void printRunUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " run <model-path|dir> [options]\n"
        << "Options:\n"
        << "  --temperature <T>  Sampling temperature (default: 0.0 = greedy)\n"
        << "  --top-k <k>        Top-k sampling (default: 0 = disabled)\n"
        << "  --seed <S>         Random seed (default: 0 = random)\n"
        << "  -n, --n-predict <n> Maximum response tokens per turn (default: "
           "64)\n"
        << "  --tokenizer <path> Path to vocab.json tokenizer file\n"
        << "  --backend cpu|cuda Execution backend (default: cpu; cuda "
           "requires "
           "-DMINI_LLAMA_CUDA=ON)\n"
        << "  --device <n>       CUDA device id for --backend cuda (default: "
           "0)\n"
        << "  -h, --help         Show this help\n";
}

static int runInspectGguf(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " inspect-gguf <path>\n";
        return 1;
    }
    GgufReader reader;
    if (!reader.load(argv[2])) {
        std::cerr << "Failed to load GGUF: " << reader.load_error << "\n";
        return 1;
    }
    inspectGguf(reader);
    return 0;
}

int runChat(int argc, char** argv) {
    if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                      std::strcmp(argv[2], "--help") == 0)) {
        printRunUsage(argv[0]);
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Missing model directory.\n";
        printRunUsage(argv[0]);
        return 1;
    }

    std::string model_dir = argv[2];
    float temperature = 0.0f;
    int top_k = 0;
    unsigned int seed = 0;
    int max_response_tokens = 64;
    std::string tokenizer_path;
    BackendConfig backend_config;

    // 解析参数
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!parseFloatArg(argv[++i], temperature)) {
                std::cerr << "Invalid --temperature value. Expected a "
                             "non-negative float.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parseIntArg(argv[++i], top_k)) {
                std::cerr << "Invalid --top-k value. Expected a non-negative "
                             "integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parseUintArg(argv[++i], seed)) {
                std::cerr << "Invalid --seed value. Expected a non-negative "
                             "integer.\n";
                return 1;
            }
        } else if ((std::strcmp(argv[i], "-n") == 0 ||
                    std::strcmp(argv[i], "--n-predict") == 0) &&
                   i + 1 < argc) {
            if (!parseIntArg(argv[++i], max_response_tokens) ||
                max_response_tokens <= 0) {
                std::cerr << "Invalid --n-predict value. Expected a positive "
                             "integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            if (!parseBackendArg(argv[++i], backend_config)) {
                std::cerr << "Invalid --backend value. Supported values: cpu, "
                             "cuda.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            if (!parseDeviceArg(argv[++i], backend_config)) {
                std::cerr << "Invalid --device value. Expected a non-negative "
                             "integer.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            printRunUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            printRunUsage(argv[0]);
            return 1;
        }
    }

    // Load model
    LoadedModel lm = loadModelAndTokenizer(model_dir, "", tokenizer_path);
    if (!lm.model.loaded) {
        std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        std::cerr << "Failed to load tokenizer.\n";
        return 1;
    }
    if (lm.model.config.vocab_size < lm.tokenizer->vocabSize()) {
        std::cerr << "Model vocab_size must be at least "
                  << lm.tokenizer->vocabSize() << " for the tokenizer.\n";
        return 1;
    }

    MiniLlamaModel& model = lm.model;
    if (!prepareCudaWeightsOrPrint(model, backend_config)) {
        return 1;
    }

    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;

    PromptBuilder builder;
    if (!lm.chat_template.empty()) {
        builder.setChatTemplate(lm.chat_template);
    }

    // 终端  +  session
    Terminal term;
    ChatSession session;

    SamplingParams sampling_params;
    sampling_params.temperature = temperature;
    sampling_params.top_k = top_k;
    sampling_params.seed = seed;
    session.setSamplingParams(sampling_params);

    // Plain text mode keeps the default system message in session state.
    if (lm.chat_template.empty()) {
        session.addMessage("system", "You are a helpful assistant.");
    }

    term.printMessage("mini-llama.cpp chat");
    term.printMessage("backend: " + backendKindName(backend_config.kind));
    term.printMessage("Type /help for commands, /exit to quit.\n");
    if (lm.chat_template.empty() && model.config.max_seq_len <= 256) {
        term.printMessage(
            "Tiny teaching model: random weights, small context window, "
            "smoke-test "
            "output.");
        term.printMessage(
            "Real chat demo: ./build/mini-llama run models/chat -n 8\n");
    }

    MiniLlamaContext ctx(&model);

    while (true) {
        term.printUserPrompt();
        std::string input = term.readLine();
        if (input.empty() && std::cin.eof()) {
            break;
        }

        // 处理特殊命令
        // Handle commands
        if (input == "/help") {
            term.printHelp();
            continue;
        }
        if (input == "/exit") {
            break;
        }
        if (input == "/clear") {
            session.clear();
            ctx = MiniLlamaContext(&model);
            if (lm.chat_template.empty()) {
                session.addMessage("system", "You are a helpful assistant.");
            }
            term.printMessage("Chat history cleared.\n");
            continue;
        }
        if (input == "/stats") {
            term.printStats(session);
            continue;
        }
        if (input == "/params") {
            term.printParams(session.getSamplingParams());
            continue;
        }
        if (!input.empty() && input[0] == '/') {
            term.printMessage("Unknown command: " + input + "\n");
            continue;
        }
        if (input.empty()) {
            continue;
        }

        RequestContext request = startRequest(
            "run", backendKindName(backend_config.kind), model_dir);

        std::vector<ChatMessage> candidate_messages = session.getMessages();
        candidate_messages.push_back({"user", input});

        auto stage_start = RequestClock::now();
        std::string prompt_text = builder.build(candidate_messages);
        request.recordEvent(
            "prompt_build", elapsedMs(stage_start), 0,
            "messages=" + std::to_string(candidate_messages.size()));

        // tokenizer: vocab -> token id
        stage_start = RequestClock::now();
        std::vector<int> tokens = tokenizer->encode(prompt_text);
        request.tokenize_ms = elapsedMs(stage_start);
        request.prompt_tokens = static_cast<int>(tokens.size());

        request.recordEvent("tokenize", request.tokenize_ms,
                            request.prompt_tokens, "prompt");

        // token过多
        if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
            request.setError("prompt uses " + std::to_string(tokens.size()) +
                             " tokens, context window is " +
                             std::to_string(model.config.max_seq_len) +
                             ". Use /clear or a shorter prompt.");
            request.finish();
            printRequestTrace(request);
            term.printMessage("Error: " + request.error + "\n");
            continue;
        }

        int max_response =
            model.config.max_seq_len - static_cast<int>(tokens.size());
        if (max_response > max_response_tokens) {
            max_response = max_response_tokens;
        }

        session.setMessages(candidate_messages);
        MiniSampler sampler(session.getSamplingParams());
        Tensor logits;

        auto start = std::chrono::steady_clock::now();

        size_t cached_prefix_len = session.longestCachedPrefix(tokens);
        size_t context_prefix_len =
            commonPrefixLength(ctx.token_history, tokens);
        size_t prefix_len = std::min(cached_prefix_len, context_prefix_len);
        if (prefix_len >= tokens.size()) {
            ctx = MiniLlamaContext(&model);
            prefix_len = 0;
        } else if (prefix_len == 0) {
            ctx = MiniLlamaContext(&model);
        } else if (prefix_len < ctx.token_history.size()) {
            ctx.token_history.resize(prefix_len);
            ctx.pos = static_cast<int>(prefix_len - 1);
        }

        // 只encode 新的token
        std::vector<int> new_prompt_tokens(
            tokens.begin() + static_cast<std::ptrdiff_t>(prefix_len),
            tokens.end());
        session.setTokenHistory(tokens);

        // Prefill only the context suffix that is missing from KV cache.
        {
            MiniBatch prefill = MiniBatch::fromTokens(
                new_prompt_tokens, static_cast<int>(prefix_len));
            stage_start = RequestClock::now();
            logits = forwardBatch(ctx, model, prefill);
            request.prefill_ms = elapsedMs(stage_start);
            request.prefill_tokens = static_cast<int>(new_prompt_tokens.size());
            request.recordEvent(
                "prefill", request.prefill_ms, request.prefill_tokens,
                "radix_hit=" + std::to_string(cached_prefix_len) +
                    ", prefix_reuse=" + std::to_string(prefix_len));
            ctx.n_prefill_tokens += static_cast<int>(new_prompt_tokens.size());
        }

        // Generate response
        std::vector<int> generated_ids;
        std::string streamed_reply;
        int generated_count = 0;

        term.printAssistantPrefix();
        try {
            for (int i = 0; i < max_response; ++i) {
                stage_start = RequestClock::now();
                // 采样
                int next_token =
                    sampler.sample(logits, session.getSamplingParams());
                if (next_token == -1) {
                    break;
                }
                float max_logit = -1e10f;
                int max_idx = -1;
                for (size_t j = 0; j < logits.data.size(); ++j) {
                    if (logits.data[j] > max_logit) {
                        max_logit = logits.data[j];
                        max_idx = static_cast<int>(j);
                    }
                }
                request.sample_ms += elapsedMs(stage_start);
                tokens.push_back(next_token);
                session.appendToken(next_token);
                ++generated_count;

                if (next_token != tokenizer->eosId()) {
                    generated_ids.push_back(next_token);
                    std::string current_reply =
                        tokenizer->decode(generated_ids);
                    if (current_reply.size() > streamed_reply.size()) {
                        term.printTokenText(
                            current_reply.substr(streamed_reply.size()));
                        term.flush();
                        streamed_reply = current_reply;
                    }
                }

                MiniBatch decode_batch = MiniBatch::single(
                    next_token, static_cast<int>(tokens.size() - 1));
                stage_start = RequestClock::now();
                logits = forwardBatch(ctx, model, decode_batch);
                double decode_ms = elapsedMs(stage_start);
                request.decode_ms += decode_ms;
                request.recordEvent("decode", decode_ms, 1,
                                    "pos=" + std::to_string(tokens.size() - 1));
                ++ctx.n_decode_tokens;
                ++request.decode_tokens;
                if (next_token == tokenizer->eosId()) {
                    break;
                }
            }
        } catch (const std::exception& e) {
            term.newLine();
            request.setError("Inference error: " + std::string(e.what()));
            request.finish();
            printRequestTrace(request);
            term.printMessage(request.error + "\n");
            continue;
        }

        // Decode and output
        std::string assistant_reply = tokenizer->decode(generated_ids);
        if (assistant_reply.size() > streamed_reply.size()) {
            term.printTokenText(assistant_reply.substr(streamed_reply.size()));
        }
        term.newLine();
        term.newLine();

        auto end = std::chrono::steady_clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        session.addMessage("assistant", assistant_reply);
        session.recordTurn(static_cast<int>(new_prompt_tokens.size()),
                           generated_count, elapsed_ms);
        request.generated_tokens = generated_count;
        request.recordEvent("sample", request.sample_ms,
                            request.generated_tokens, "generated_tokens");
        session.recordPrefix(session.getTokenHistory());
        request.finish();
        printRequestTrace(request);
    }

    term.printMessage("Goodbye.\n");
    return 0;
}

int runInspect(const std::vector<std::string>& args) {
    // TODO: print model manifest.

    return 1;
}

// ---------------------------------------------------------------------------
// Bench mode
// ---------------------------------------------------------------------------
static void printBenchUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " bench <model-path|dir> [options]\n"
        << "Options:\n"
        << "  -p, --prompt <str>    Input prompt text (default: \"hello\")\n"
        << "  -n, --n-predict <n>   Number of tokens to generate (default: "
           "64)\n"
        << "  --seed <S>            Random seed (default: 0 = random)\n"
        << "  --tokenizer <path>    Path to vocab.json tokenizer file\n"
        << "  --quant q8_0|q4_0     Quantize loaded Linear weights before "
           "benchmark\n"
        << "  --threads <n>         Number of threads for parallel ops (0 = "
           "auto)\n"
        << "  --backend cpu|cuda    Execution backend (default: cpu; cuda "
           "requires -DMINI_LLAMA_CUDA=ON)\n"
        << "  --device <n>          CUDA device id for --backend cuda "
           "(default: "
           "0)\n"
        << "  --verbose             Print debug dumps after each step\n"
        << "  -h, --help            Show this help\n"
        << "\n"
        << "CUDA benchmark metrics:\n"
        << "  uploaded weights: CUDA-resident Linear weights loaded before "
           "inference\n"
        << "  cuda Linear/activation/attention calls: GPU kernel coverage "
           "during "
           "forward\n"
        << "  cpu attention fallback calls: attention steps that returned to "
           "CPU\n"
        << "  host->device / device->host copies: runtime transfer count and "
           "bytes\n"
        << "  cuda fallback: unsupported or missing quantized Linear weights "
           "running on CPU\n";
}

static double benchmarkRepeated(int warmup, int iterations,
                                const std::function<void()>& fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return elapsed_ms / static_cast<double>(iterations);
}

static int runBench(int argc, char** argv) {
    if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                      std::strcmp(argv[2], "--help") == 0)) {
        printBenchUsage(argv[0]);
        return 0;
    }

    if (argc < 3) {
        std::cerr << "Missing model directory.\n";
        printBenchUsage(argv[0]);
        return 1;
    }

    std::string model_dir = argv[2];
    std::string prompt = "hello";
    int n_predict = 64;
    unsigned int seed = 0;
    std::string tokenizer_path;
    bool verbose = false;
    std::string quant_type;
    int n_threads = 0;

    for (int i = 3; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-p") == 0 ||
             std::strcmp(argv[i], "--prompt") == 0) &&
            i + 1 < argc) {
            prompt = argv[++i];
        } else if ((std::strcmp(argv[i], "-n") == 0 ||
                    std::strcmp(argv[i], "--n-predict") == 0) &&
                   i + 1 < argc) {
            if (!parseIntArg(argv[++i], n_predict)) {
                std::cerr << "Invalid --n-predict value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parseUintArg(argv[++i], seed)) {
                std::cerr << "Invalid --seed value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            quant_type = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            if (!parseIntArg(argv[++i], n_threads) || n_threads < 0) {
                std::cerr << "Invalid --threads value.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            printBenchUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            printBenchUsage(argv[0]);
            return 1;
        }
    }

    if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
        std::cerr << "Invalid --quant value: " << quant_type
                  << ". Supported values: q8_0, q4_0.\n";
        return 1;
    }

    LoadedModel lm = loadModelAndTokenizer(model_dir, "", tokenizer_path);
    if (!lm.model.loaded) {
        std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
        return 1;
    }
    if (!lm.tokenizer) {
        std::cerr << "Failed to load tokenizer.\n";
        return 1;
    }
    if (lm.model.config.vocab_size < lm.tokenizer->vocabSize()) {
        std::cerr << "Model vocab_size must be at least "
                  << lm.tokenizer->vocabSize() << " for the tokenizer.\n";
        return 1;
    }

    MiniLlamaModel baseline_model = lm.model;
    MiniLlamaModel& model = lm.model;
    std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;
    std::vector<int> tokens = tokenizer->encode(prompt);
    if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
        std::cerr << "Prompt too long.\n";
        return 1;
    }
    if (tokens.size() + static_cast<size_t>(n_predict) >
        static_cast<size_t>(model.config.max_seq_len)) {
        n_predict = model.config.max_seq_len - static_cast<int>(tokens.size());
    }

    std::cout << "  prompt: \"" << prompt << "\" (" << tokens.size()
              << " tokens)\n";
    ThreadPool::setThreadCount(n_threads);
    std::cout << "  n_predict: " << n_predict << "\n";
    std::cout << "  seed: " << seed << "\n";
    std::cout << "  quant: "
              << (quant_type.empty() ? "model-native" : quant_type) << "\n";

    BenchmarkResult result =
        runBenchmark(model, tokens, n_predict, seed, verbose);

    std::cout << "  threads: " << ThreadPool::getThreadCount() << "\n";
    std::cout << "  verbose: " << (verbose ? "true" : "false") << "\n\n";
    std::cout << "Results:\n";
    std::cout << "  prompt tokens:     " << result.n_prompt_tokens << "\n";
    std::cout << "  generated tokens:  " << result.n_generated_tokens << "\n";
    std::cout << "  Decode tokens:     " << result.n_decode_tokens << "\n";
    std::cout << "  prefill time:      " << std::fixed << std::setprecision(2)
              << result.prefill_ms << " ms\n";
    std::cout << "  Decode time:       " << std::fixed << std::setprecision(2)
              << result.decode_ms << " ms\n";
    std::cout << "  total time:        " << std::fixed << std::setprecision(2)
              << (result.prefill_ms + result.decode_ms) << " ms\n";
    std::cout << "  tokens/s (total):  " << std::fixed << std::setprecision(2)
              << result.tokensPerSec() << "\n";
    std::cout << "  tokens/s (Decode): " << std::fixed << std::setprecision(2)
              << result.decodeTokensPerSec() << "\n";

    // Memory footprint
    size_t actual_bytes = modelWeightBytes(model);
    size_t f32_bytes = modelWeightBytesF32(model);
    std::cout << "\n  weight memory:\n";
    std::cout << "    actual:    " << actual_bytes << " bytes (" << std::fixed
              << std::setprecision(2) << (actual_bytes / (1024.0 * 1024.0))
              << " MB)\n";
    std::cout << "    f32 equiv: " << f32_bytes << " bytes (" << std::fixed
              << std::setprecision(2) << (f32_bytes / (1024.0 * 1024.0))
              << " MB)\n";
    std::cout << "    savings:   " << std::fixed << std::setprecision(2)
              << (static_cast<double>(f32_bytes) / actual_bytes)
              << "x compression\n";

    return 0;
}

void printUsage() {
    std::cout << "mini-llama <command> [options]\n"
                 "\n"
                 "Commands:\n"
                 "  generate   single-shot generation\n"
                 "  run        interactive chat\n"
                 "  inspect    print model manifest\n"
                 "  bench      prefill/decode timing\n";
}

}  // namespace mini_llama

int main(int argc, char** argv) {
    // 初始化文件日志（异步 + 控制台彩色 + 按天切割），并加载错误码配置。
    // 无参 LoadFromFile 使用编译期注入的绝对路径，不依赖运行时工作目录。
    errlog::LogGuard log_guard;

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) {
        mini_llama::printUsage();
        return 1;
    }

    const std::string command = args[0];
    std::vector<std::string> rest(args.begin() + 1, args.end());

    int rc = 1;
    if (command == "generate") {
        rc = mini_llama::runGenerate(argc, argv);
    } else if (command == "run") {
        rc = mini_llama::runChat(argc, argv);
    } else if (command == "inspect-gguf") {
        rc = mini_llama::runInspectGguf(argc, argv);
    } else if (command == "bench") {
        rc = mini_llama::runBench(argc, argv);
    } else if (!command.empty() && command[0] == '-') {
        // Backward compatibility: default to generate mode when first arg is a
        // flag. Shift argv[1..] down by inserting "generate" at position 1.
        std::vector<char*> shifted_argv;
        shifted_argv.reserve(argc + 1);
        shifted_argv.push_back(argv[0]);
        shifted_argv.push_back(const_cast<char*>("generate"));
        for (int i = 1; i < argc; ++i) {
            shifted_argv.push_back(argv[i]);
        }
        return mini_llama::runGenerate(argc + 1, shifted_argv.data());
    } else {
        mini_llama::printUsage();
    }

    return rc;
}
