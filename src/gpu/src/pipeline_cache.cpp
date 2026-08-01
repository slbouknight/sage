#include <sage/gpu/pipeline_cache.hpp>

#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/vk_check.hpp>

#include <cstring>
#include <fstream>
#include <utility>
#include <vector>
#include <cstddef>
#include <system_error>

namespace sage::gpu {

namespace {

// The first bytes of a cache blob are a VkPipelineCacheHeaderVersionOne. A blob
// from another driver or GPU is useless to us and must not be handed to
// vkCreatePipelineCache, so validate before trusting it.
bool header_matches(const std::vector<std::byte>& data, const VkPhysicalDeviceProperties& props) {
    if (data.size() < sizeof(VkPipelineCacheHeaderVersionOne)) {
        return false;
    }

    VkPipelineCacheHeaderVersionOne header{};
    std::memcpy(&header, data.data(), sizeof(header));

    return header.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
           header.vendorID == props.vendorID && header.deviceID == props.deviceID &&
           std::memcmp(header.pipelineCacheUUID, props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return {};
    }

    std::vector<std::byte> data(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), size);
    if (!file.good()) {
        return {};
    }

    return data;
}

}  // namespace

PipelineCache::PipelineCache(const Device& device, std::filesystem::path path)
    : device_(device), path_(std::move(path)) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device_.physical_device(), &props);

    const std::vector<std::byte> data = read_file(path_);

    VkPipelineCacheCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

    if (data.empty()) {
        SAGE_LOG_DEBUG("No pipeline cache at {}; compiling cold", path_.string());
    } else if (header_matches(data, props)) {
        info.initialDataSize = data.size();
        info.pInitialData = data.data();
        SAGE_LOG_INFO("Pipeline cache loaded: {} bytes", data.size());
    } else {
        SAGE_LOG_INFO("Pipeline cache rejected (different driver or GPU); compiling cold");
    }

    VK_CHECK(vkCreatePipelineCache(device_.handle(), &info, nullptr, &cache_));
}

void PipelineCache::save() const {
    std::size_t size = 0;
    if (vkGetPipelineCacheData(device_.handle(), cache_, &size, nullptr) != VK_SUCCESS ||
        size == 0) {
        SAGE_LOG_WARN("Could not query pipeline cache size; not saving");
        return;
    }

    std::vector<std::byte> data(size);
    if (vkGetPipelineCacheData(device_.handle(), cache_, &size, data.data()) != VK_SUCCESS) {
        SAGE_LOG_WARN("Could not read pipeline cache data; not saving");
        return;
    }
    data.resize(size);

    std::error_code ec;
    std::filesystem::create_directories(path_.parent_path(), ec);
    if (ec) {
        SAGE_LOG_WARN("Could not create pipeline cache directory: {}", ec.message());
        return;
    }

    std::ofstream file(path_, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        SAGE_LOG_WARN("Could not open pipeline cache for writing: {}", path_.string());
        return;
    }

    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    if (!file.good()) {
        SAGE_LOG_WARN("Failed to write pipeline cache: {}", path_.string());
        return;
    }

    SAGE_LOG_INFO("Pipeline cache saved: {} bytes", data.size());
}

PipelineCache::~PipelineCache() {
    if (cache_ != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(device_.handle(), cache_, nullptr);
    }
}

} // namespace sage::gpu