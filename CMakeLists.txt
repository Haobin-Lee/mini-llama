cmake_minimum_required(VERSION 3.17)
project(mini-llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# --- Core library sources (MVP: CPU, F32) ---
set(MINI_LLAMA_CORE_SOURCES
    src/tensor.cc
)

# --- Main executable ---
add_executable(mini-llama ${MINI_LLAMA_CORE_SOURCES} src/main.cc)

# --- Test executable ---
add_executable(mini-llama-tests
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_main.cc
    tests/test_tensor.cc
)

enable_testing()
add_test(NAME mini-llama-tests COMMAND mini-llama-tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
