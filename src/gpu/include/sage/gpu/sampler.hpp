#pragma once

#include <vulkan/vulkan.h>

namespace sage::gpu {

class Device;

// One sampler shared by every texture. glTF can specify per-texture filter and
// wrap modes. Sharing one sampler keeps every combined image-sampler descriptor in
// the bindless array identical.
class Sampler {
public:
    explicit Sampler(const Device& device);
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) = delete;
    Sampler& operator=(Sampler&&) = delete;

    [[nodiscard]] VkSampler handle() const { return sampler_; }

private:
    const Device& device_;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

}  // namespace sage::gpu