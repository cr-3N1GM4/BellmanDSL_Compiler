// test_framework.hpp — Micro test framework for MDP-DSL v2.0
// No external dependencies. Provides ASSERT_TRUE, ASSERT_EQ, ASSERT_NEAR macros.
#pragma once
#include <iostream>
#include <string>
#include <cmath>

namespace tf {
    inline int passes = 0;
    inline int failures = 0;
}

#define ASSERT_TRUE(expr) \
    do { if (!(expr)) { \
        std::cerr << "  [FAIL] " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        ++tf::failures; return; } ++tf::passes; } while(0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { \
        std::cerr << "  [FAIL] " << #a << " == " << #b \
                  << " -> got '" << (a) << "' != '" << (b) << "'" \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        ++tf::failures; return; } ++tf::passes; } while(0)

#define ASSERT_NEAR(a, b, tol) \
    do { if (std::abs((double)(a)-(double)(b)) > (tol)) { \
        std::cerr << "  [FAIL] |" << #a << " - " << #b << "| <= " << (tol) \
                  << " -> |" << (a) << " - " << (b) << "| = " << std::abs((double)(a)-(double)(b)) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        ++tf::failures; return; } ++tf::passes; } while(0)

#define TEST(name) void name()
#define RUN_TEST(name) do { std::cout << "  " << #name << "... "; name(); \
    std::cout << "ok\n"; } while(0)
