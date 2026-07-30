#pragma once

#include <spdlog/spdlog.h>

#include <memory>

namespace sage::core::log {

// Must be called once, before any SAGE_LOG_* use, typically first thing in
// main().
void init();

std::shared_ptr<spdlog::logger>& logger();

}  // namespace sage::core::log

#define SAGE_LOG_TRACE(...) SPDLOG_LOGGER_TRACE(::sage::core::log::logger(), __VA_ARGS__)
#define SAGE_LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(::sage::core::log::logger(), __VA_ARGS__)
#define SAGE_LOG_INFO(...) SPDLOG_LOGGER_INFO(::sage::core::log::logger(), __VA_ARGS__)
#define SAGE_LOG_WARN(...) SPDLOG_LOGGER_WARN(::sage::core::log::logger(), __VA_ARGS__)
#define SAGE_LOG_ERROR(...) SPDLOG_LOGGER_ERROR(::sage::core::log::logger(), __VA_ARGS__)
#define SAGE_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(::sage::core::log::logger(), __VA_ARGS__)
