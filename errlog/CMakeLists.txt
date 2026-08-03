# errlog 模块独立构建脚本。
#
# 特点：
#   1. 编译前自动执行 Python 脚本，由 error_code_config.json 生成
#      generated/error_code_enum.h（generated 头文件不手写、不入库亦可）。
#   2. 仅依赖 spdlog、nlohmann/json 两个第三方库。
#
# 既可被根 CMakeLists 通过 add_subdirectory(errlog) 引入，
# 也可单独 configure 本目录进行构建。

cmake_minimum_required(VERSION 3.17)

# 独立构建（本目录作为顶层工程）时才声明 project 与 C++ 标准；
# 被根 CMakeLists 通过 add_subdirectory 引入时，直接沿用根工程的配置。
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(errlog CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# ---- 第三方依赖（幂等，standalone 构建时也需要）----
find_package(spdlog REQUIRED)      # 传递依赖会自动带上 fmt
find_package(nlohmann_json REQUIRED)

# ---- 代码生成：configure 阶段先跑一次，保证首次构建即有头文件 ----
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(ERRLOG_DIR       ${CMAKE_CURRENT_SOURCE_DIR})
set(ERRLOG_CONFIG    ${ERRLOG_DIR}/error_code_config.json)
set(ERRLOG_GEN_PY    ${ERRLOG_DIR}/gen_error_enum.py)
set(ERRLOG_GEN_HEADER ${ERRLOG_DIR}/generated/error_code_enum.h)

# 自定义命令：JSON 或脚本变化时，重新生成枚举头文件。
add_custom_command(
    OUTPUT ${ERRLOG_GEN_HEADER}
    COMMAND ${Python3_EXECUTABLE} ${ERRLOG_GEN_PY}
            --config ${ERRLOG_CONFIG}
            --output ${ERRLOG_GEN_HEADER}
    DEPENDS ${ERRLOG_CONFIG} ${ERRLOG_GEN_PY}
    COMMENT "[errlog] generating error_code_enum.h from JSON config"
    VERBATIM)

add_custom_target(errlog_gen_enum DEPENDS ${ERRLOG_GEN_HEADER})

# ---- errlog 静态库 ----
add_library(errlog STATIC
    ${ERRLOG_DIR}/error_code_mgr.cc
    ${ERRLOG_DIR}/log_init.cc
    ${ERRLOG_DIR}/log_guard.cc
)
add_dependencies(errlog errlog_gen_enum)

# 头文件引用形如 "errlog/xxx.h"（相对 errlog 父目录）与
# "generated/xxx.h"（相对 errlog 目录），故两个根目录都要加入包含路径。
target_include_directories(errlog PUBLIC
    ${ERRLOG_DIR}/..
    ${ERRLOG_DIR}
)

target_link_libraries(errlog PUBLIC
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

# 将配置文件的绝对路径注入为编译宏，作为 LoadFromFile() 的默认路径，
# 这样无论从哪个工作目录启动可执行程序都能找到 JSON（可用环境变量
# MINI_LLAMA_ERROR_CONFIG 覆盖）。
target_compile_definitions(errlog PUBLIC
    ERRLOG_DEFAULT_CONFIG_PATH="${ERRLOG_CONFIG}")
