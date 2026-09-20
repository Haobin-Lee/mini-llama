# mini-llama

一个用 C++17 从零实现的轻量级 LLM 推理引擎（Llama / Qwen2 风格），支持 CPU 与 CUDA 双后端、GGUF 与量化模型、分页 KV cache 与批量推理。

## 特性

- **双后端**：CPU与 CUDA，通过 `backend.h` 统一抽象。
- **多种模型格式**：
  - 旧式 manifest：`model.json` + `model.bin`（如 `models/tiny/`）
  - GGUF：支持 `q8_0` / `q4_0` / `q4_1` 量化权重
- **量化**：`q8_0` / `q4_0` / `q4_1`，运行时也可通过 `--quant` 即时量化。
- **分页 KV cache**：`kv_cache` + `radix_tree` + `node_memory_pool`，支持前缀复用（prefix reuse）。
- **批量推理**：`batch` + `threadpool` + `matmul_dispatch` 的 prefill/decode 流水线。
- **采样**：`temperature`、`top-k`、可复现的 `seed`。
- **交互式聊天**：流式输出、chat template 支持（从 GGUF MetaData读取）。
- **日志框架**：spdlog，按天切分到 `llama_logs/`，错误码由 `errlog/error_code_config.json` 在构建期生成。

## 构建

### 依赖

- CMake ≥ 3.17
- 支持 C++17 的编译器（如果nvcc版本为11.x，CUDA构建可以使用 `gcc-9`/`g++-9` 作为 host compiler）
- CUDA Toolkit（仅当启用 CUDA 后端时）
- Python3

### CPU 构建

`MINI_LLAMA_CUDA` 默认 **ON**，纯 CPU 机器需要显式关闭：

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF
cmake --build build -j16
```

CPU 开发循环可用 AVX2 + `-ffast-math`：

```bash
cmake -DMINI_LLAMA_AVX2=ON -DCMAKE_CXX_FLAGS="-ffast-math" -B build
cmake --build build -j16
```

### CUDA 构建（RTX2060 super）

CUDA 构建： `gcc-9` 工具链与 `CMAKE_CUDA_ARCHITECTURES=75`（见 `cuda_build.sh`）：

```bash
cmake -DMINI_LLAMA_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_C_COMPILER=gcc-9 \
  -DCMAKE_CXX_COMPILER=g++-9 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-9 -B build
cmake --build build -j16
```

构建产物：

- `build/mini-llama` —— CLI 主程序
- `build/mini-llama-tests` —— 测试程序

## 使用

CLI 提供以下子命令（裸 flag 参数会回退到 `generate`）：

```text
mini-llama <command> [options]

Commands:
  generate       单次生成
  run            交互式聊天
  inspect-gguf   查看 GGUF 元数据
  bench          prefill/decode 计时
```

### 单次生成

```bash
./build/mini-llama generate \
  --model models/tiny/model.bin \
  --config models/tiny/model.json \
  -p "hello" -n 16 \
  --temperature 0.0 --top-k 0 --seed 1 \
  --threads 4
```

常用选项：`--model <path|dir>`、`--config <path>`、`-p/--prompt`、`-n/--n-predict`、`--temperature`、`--top-k`、`--seed`、`--tokenizer`、`--quant q8_0|q4_0`、`--threads`、`--backend cpu|cuda`、`--device <n>`、`--dump-logits <dir>`。

### 交互式聊天

```bash
./build/mini-llama run models/chat -n 8
```

> `models/tiny/` 是随机权重的模型，仅用于冒烟测试；真实聊天请使用 `models/chat` 中的 Qwen2 量化 GGUF。

### 查看 GGUF

```bash
./build/mini-llama inspect-gguf <path-to.gguf>
```

### 基准测试

```bash
./build/mini-llama bench models/chat -p hi -n 15 --threads 4 --seed 1
```

## 测试

项目使用自定义零依赖测试框架，测试返回 `bool` 并用 `MINI_LLAMA_ASSERT_*` 宏断言。

```bash
# 二选一
ctest --test-dir build
./build/mini-llama-tests
```

> 测试命令必须在仓库根目录运行，请勿移动或重命名 `models/`。

`golden-logits` 测试在 `models/tiny/` 上对比 C++ `--dump-logits` 输出与 numpy 参考实现（`scripts/test_golden.py`），需要 numpy。可用 `make_goken.sh` 重新生成 golden 数据。

## 项目结构

```text
include/mini_llama/   头文件
src/                  实现文件（所有核心源文件同时编译进主程序与测试）
errlog/               错误码 + 文件日志框架
models/tiny/          随机权重的测试模型（JSON/BIN 格式）
models/chat/          Qwen2 量化 GGUF（git 忽略）
scripts/              golden 数据生成与对比脚本
tests/                自定义测试框架用例
docs/                 审计记录与知识点笔记
```

## 许可

MIT License
