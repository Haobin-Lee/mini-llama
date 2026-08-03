// Copyright (c) 2026
// SPDX-License-Identifier: MIT

#ifndef TESTS_TEST_MAIN_H_
#define TESTS_TEST_MAIN_H_

// Minimal zero-dependency test runner (declarations + assertion macros).

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

struct TestCase {
    std::string name;
    std::function<bool()> fn;
};

std::vector<TestCase>& testRegistry();
void registerTest(const std::string& name, std::function<bool()> fn);

// Register a test with: TEST(my_test_name) { ...; return true; }
#define TEST(name)                                       \
    static bool name();                                  \
    namespace {                                          \
    struct Register_##name {                             \
        Register_##name() { registerTest(#name, name); } \
    } register_##name;                                   \
    }                                                    \
    static bool name()

inline bool assertTrue(bool cond, const char* expr, const char* file,
                       int line) {
    if (!cond) {
        std::cerr << "  ASSERTION FAILED: " << expr << " at " << file << ":"
                  << line << std::endl;
        return false;
    }
    return true;
}

inline bool assertEqFloat(float a, float b, float tol, const char* expr_a,
                          const char* expr_b, const char* file, int line) {
    if (std::fabs(a - b) > tol) {
        std::cerr << "  ASSERTION FAILED: |" << expr_a << " - " << expr_b
                  << "| = |" << a << " - " << b << "| = " << std::fabs(a - b)
                  << " > " << tol << " at " << file << ":" << line << std::endl;
        return false;
    }
    return true;
}

#define MINI_LLAMA_ASSERT_TRUE(cond)                          \
    do {                                                      \
        if (!assertTrue((cond), #cond, __FILE__, __LINE__)) { \
            return false;                                     \
        }                                                     \
    } while (0)
#define MINI_LLAMA_ASSERT_EQ(a, b)                                       \
    do {                                                                 \
        if (!assertTrue((a) == (b), #a " == " #b, __FILE__, __LINE__)) { \
            return false;                                                \
        }                                                                \
    } while (0)
#define MINI_LLAMA_ASSERT_NEAR(a, b, tol)                                  \
    do {                                                                   \
        if (!assertEqFloat((a), (b), (tol), #a, #b, __FILE__, __LINE__)) { \
            return false;                                                  \
        }                                                                  \
    } while (0)
#define MINI_LLAMA_ASSERT_FAIL(msg)                                      \
    do {                                                                 \
        std::cerr << "  ASSERTION FAILED: " << msg << " at " << __FILE__ \
                  << ":" << __LINE__ << std::endl;                       \
        return false;                                                    \
    } while (0)

#endif  // TESTS_TEST_MAIN_H_
