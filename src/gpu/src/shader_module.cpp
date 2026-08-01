#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/shader_module.hpp>
#include <sage/gpu/vk_check.hpp>

#include <cstdint>
#include <fstream>
#include <vector>

namespace sage::gpu {

namespace {
std::vector<std::uint32_t> read_spirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        SAGE_LOG_ERROR("Failed to open SPIR-V file: {}", path.string());
    }
    SAGE_VERIFY(file.is_open(), "Failed to open SPIR-V file (path logged above)");

    const std::streamsize size = file.tellg();
    SAGE_VERIFY(size > 0 && size % 4 == 0,
                "SPIR-V file size must be a positive multiple of 4 bytes");

    std::vector<std::uint32_t> code(static_cast<std::size_t>(size) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), size);
    SAGE_VERIFY(file.good(), "Failed to read SPIR-V file");

    return code;
}

}  // namespace

ShaderModule::ShaderModule(const Device& device, const std::filesystem::path& spirv_path)
    : device_(device) {
    const std::vector<std::uint32_t> code = read_spirv(spirv_path);

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size() * sizeof(std::uint32_t);
    info.pCode = code.data();

    VK_CHECK(vkCreateShaderModule(device_.handle(), &info, nullptr, &module_));
    SAGE_LOG_DEBUG("Loaded shader module: {}", spirv_path.string());
}

ShaderModule::~ShaderModule() {
    if (module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_.handle(), module_, nullptr);
    }
}
}  // namespace sage::gpu