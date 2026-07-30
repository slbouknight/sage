#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>

namespace sage::core::log {

namespace {
std::shared_ptr<spdlog::logger> g_logger;  // NOLINT(cert-err58-cpp)
}

void init() {
    if (g_logger) {
        return;
    }
    g_logger = spdlog::stdout_color_mt("sage");
    g_logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    g_logger->set_level(spdlog::level::trace);
    g_logger->flush_on(spdlog::level::warn);
}

std::shared_ptr<spdlog::logger>& logger() {
    SAGE_VERIFY(g_logger != nullptr, "sage::core::log::init() must run before any SAGE_LOG_* call");
    return g_logger;
}

}  // namespace sage::core::log
