cmake_minimum_required(VERSION 3.17)
project(mini-llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

option(MINI_LLAMA_AVX2 "Enable AVX2 SIMD optimizations" ON)


if(MINI_LLAMA_AVX2)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        add_compile_options(-mavx2 -mfma)
        message(STATUS "AVX2 + FMA enabled")
    else()
        message(WARNING "AVX2 not supported by compiler, falling back to scalar")
    endif()
endif()


include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# --- 错误码 + 文件日志框架 ---
# 构建 errlog 静态库，并在编译前自动由 error_code_config.json 生成
# generated/error_code_enum.h。errlog PUBLIC 传递 spdlog/json 及其头文件路径。
add_subdirectory(errlog)

# 编译后把错误码配置拷贝到“可执行程序同级的 errlog/ 目录”，
# 使程序无论从哪个工作目录启动，都能在自身目录下就近找到配置文件。
set(ERRLOG_CONFIG_SRC ${CMAKE_SOURCE_DIR}/errlog/error_code_config.json)
function(errlog_copy_config target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:${target}>/errlog
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ERRLOG_CONFIG_SRC}
                $<TARGET_FILE_DIR:${target}>/errlog/error_code_config.json
        COMMENT "[${target}] copy error_code_config.json next to executable"
        VERBATIM)
endfunction()

# --- Core library sources (MVP: CPU, F32) ---
set(MINI_LLAMA_CORE_SOURCES
    src/tensor.cc
    src/tokenizer_bpe.cc
    src/tokenizer_ascii.cc
    src/tokenizer_json.cc
    src/ops.cc
    src/matmul_dispatch.cc
    src/kv_cache.cc
    src/threadpool.cc
    src/forward.cc
    src/request_context.cc
    src/quantized_tensor.cc
    src/context.cc
    src/model.cc
    src/chat.cc
    src/sampler.cc
    src/terminal.cc
    src/loader.cc
    src/prompt_builder.cc
    src/node_memory_pool.cc
    src/radix_tree.cc
    src/batch.cc
    src/gguf.cc
    src/gguf_loader.cc
    src/gguf_tokenizer.cc
    src/debug.cc
    src/quant.cc
)

# --- Main executable ---
add_executable(mini-llama ${MINI_LLAMA_CORE_SOURCES}
    src/main.cc
)
# 链接 errlog（自动带上 spdlog/json 头文件路径与库）。
target_link_libraries(mini-llama PRIVATE errlog)
# 保证枚举头文件先于核心源码生成。
add_dependencies(mini-llama errlog_gen_enum)
# 拷贝配置到可执行程序同级 errlog/ 目录。
errlog_copy_config(mini-llama)
# --- Test executable ---
add_executable(mini-llama-tests
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_main.cc
    tests/test_tensor.cc
    tests/test_tokenizer.cc
    tests/test_ops.cc
    tests/test_kv_cache.cc
    tests/test_forward.cc
    tests/test_batch.cc
    tests/test_quant.cc
)
target_link_libraries(mini-llama-tests PRIVATE errlog)
add_dependencies(mini-llama-tests errlog_gen_enum)
# 拷贝配置到测试可执行程序同级 errlog/ 目录。
errlog_copy_config(mini-llama-tests)

enable_testing()
add_test(NAME mini-llama-tests COMMAND mini-llama-tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(GTest REQUIRED)
target_link_libraries(mini-llama-tests PRIVATE GTest::gtest GTest::gtest_main)


if(Python3_Interpreter_FOUND)
    add_test(
        NAME golden-logits
        COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_golden.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()