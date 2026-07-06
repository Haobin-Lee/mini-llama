// Copyright (c) 2026
// SPDX-License-Identifier: MIT
//
// Test runner.
// Add tests in the test_*.cc files using the TEST(name) macro.

#include "tests/test_main.h"

std::vector<TestCase>& testRegistry() {
    static std::vector<TestCase> registry;
    return registry;
}

void registerTest(const std::string& name, std::function<bool()> fn) {
    testRegistry().push_back({name, std::move(fn)});
}

int main() {
    int passed = 0;
    int failed = 0;
    for (const auto& tc : testRegistry()) {
        bool ok = false;
        try {
            ok = tc.fn();
        } catch (const std::exception& e) {
            std::cerr << "[FAIL] " << tc.name << " threw: " << e.what()
                      << std::endl;
            ok = false;
        }
        if (ok) {
            std::cout << "[PASS] " << tc.name << std::endl;
            ++passed;
        } else {
            std::cerr << "[FAIL] " << tc.name << std::endl;
            ++failed;
        }
    }
    std::cout << "\n"
              << passed << " passed, " << failed << " failed, "
              << testRegistry().size() << " total" << std::endl;
    return failed == 0 ? 0 : 1;
}
