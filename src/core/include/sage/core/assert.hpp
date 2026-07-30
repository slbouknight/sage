#pragma once

#include <cstdio>
#include <cstdlib>

namespace sage::core {

// Deliberately independent of sage::core::log: assertions must still fire
// (and be visible) if they trip during logger init or before it's run.
[[noreturn]] inline void assert_fail(const char* expr, const char* file, int line,
                                     const char* msg) {
    std::fprintf(stderr, "SAGE_ASSERT failed: %s\n    at %s:%d\n    %s\n", expr, file, line, msg);
    std::fflush(stderr);
#if defined(__GNUC__) || defined(__clang__)
    __builtin_trap();
#else
    std::abort();
#endif
}

}  // namespace sage::core

// SAGE_VERIFY always fires, release builds included — use for conditions
// that must never be false regardless of build config (e.g. init ordering).
#define SAGE_VERIFY(cond, msg)                                           \
    do {                                                                 \
        if (!(cond)) {                                                   \
            ::sage::core::assert_fail(#cond, __FILE__, __LINE__, (msg)); \
        }                                                                \
    } while (0)

// SAGE_ASSERT compiles out under NDEBUG — use for hot-path invariants.
#if defined(NDEBUG)
#define SAGE_ASSERT(cond, msg) ((void)0)
#else
#define SAGE_ASSERT(cond, msg) SAGE_VERIFY(cond, msg)
#endif
