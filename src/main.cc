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
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/debug.h"
#include "mini_llama/forward.h"
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

static Tensor runLogitsForTokens(const MiniLlamaModel& model,
                                 const std::vector<int>& tokens) {
    MiniLlamaContext ctx(&model);
    MiniBatch batch = MiniBatch::fromTokens(tokens, 0);
    return forwardBatch(ctx, model, batch);
}

static std::string ResolveManifestTokenizerPath(
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
    }

    // Directory-based JSON+BIN format
    if (std::filesystem::is_directory(path)) {
        model_path = path + "/model.bin";
        config_path = path + "/model.json";
    } else {
        model_path = path;
        if (!explicit_config_path.empty()) {
            config_path = explicit_config_path;
        } else {
            config_path = std::filesystem::path(path).parent_path().string() +
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
        tokenizer_path = ResolveManifestTokenizerPath(config_path);
    }
    result.tokenizer = std::make_unique<JsonVocabTokenizer>(tokenizer_path);

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

int runGenerate(const std::vector<std::string>& args) {
    // TODO: single-shot generation.
    return 1;
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

        RequestContext request = startRequest("run", "cpu", model_dir);

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
                request.sample_ms += elapsedMs(stage_start);
                tokens.push_back(next_token);
                session.appendToken(next_token);
                ++generated_count;

                if (next_token != tokenizer->eosId()) {
                    generated_ids.push_back(next_token);
                    std::string current_reply =
                        tokenizer->decode(generated_ids);
                    // if (current_reply.size() > streamed_reply.size()) {
                    term.printTokenText(
                        current_reply.substr(streamed_reply.size()));
                    term.flush();
                    //     streamed_reply = current_reply;
                    // }
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

int runBench(const std::vector<std::string>& args) {
    // TODO: prefill/decode timing.

    return 1;
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
    errlog::LogGuard logGuard;

    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) {
        mini_llama::printUsage();
        return 1;
    }

    const std::string command = args[0];
    std::vector<std::string> rest(args.begin() + 1, args.end());

    int rc = 1;
    if (command == "generate") {
        // rc = runGenerate(argc, argv);
    } else if (command == "run") {
        rc = mini_llama::runChat(argc, argv);
    } else if (command == "inspect") {
        //  rc = runInspect(argc, argv);
    } else if (command == "bench") {
        // rc = runBench(argc, argv);
    } else {
        mini_llama::printUsage();
    }

    return rc;
}
