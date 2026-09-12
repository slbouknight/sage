#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace sage::gpu {

class Device;

// One descriptor set for the whole renderer. Resources are registered into
// large arrays and referenced from shaders by index (delivered via push constant)
// rather than being rebound per draw.

class BindlessSet {
public:
    // Binding numbers, as seen by shader source.
    static constexpr std::uint32_t k_storage_buffer_binding = 0;
    static constexpr std::uint32_t k_sampled_image_binding = 1;
    // The object-id attachment, read by the outline pass. Separate from the
    // array above because its sampled type is uint, not float: a shader's
    // declared type must match the format's numeric type, so it cannot share a
    // binding with the colour textures. Read with Load() rather than a sampler,
    // which is why this is SAMPLED_IMAGE and not COMBINED_IMAGE_SAMPLER --
    // integer formats support no filtering, so there is nothing for one to do.
    static constexpr std::uint32_t k_object_id_binding = 2;
    // The HDR colour target, read by the tonemap pass. Its format is float, so
    // unlike the id image above it *could* share the array at binding 1 -- but
    // that array is TextureRegistry's to allocate out of, and a render target is
    // not scene content the registry owns and destroys on reset(). Its own
    // binding keeps the two lifetimes from touching.
    static constexpr std::uint32_t k_hdr_color_binding = 3;
    // The directional light's shadow map. COMBINED_IMAGE_SAMPLER rather than
    // SAMPLED_IMAGE like the two above, because its sampler is not
    // interchangeable with theirs: it has compareEnable set, which turns a
    // sample into a depth test whose result is then filtered. Pairing the view
    // with that specific sampler is the point.
    static constexpr std::uint32_t k_shadow_map_binding = 4;
    // The tonemapped image, read by the anti-aliasing pass. Like the shadow
    // map, COMBINED_IMAGE_SAMPLER rather than SAMPLED_IMAGE: FXAA samples
    // between texel centres and depends on the bilinear blend, so the view has
    // to travel with a LINEAR sampler rather than being read by integer texel.
    static constexpr std::uint32_t k_ldr_color_binding = 5;

    // Array capacities. Far below what target hardware (RTX 5070) allows
    // (~1M update-after-bind descriptors); raise when something needs it
    static constexpr std::uint32_t k_max_storage_buffers = 1024;
    static constexpr std::uint32_t k_max_sampled_images = 1024;

    explicit BindlessSet(const Device& device);
    ~BindlessSet();

    BindlessSet(const BindlessSet&) = delete;
    BindlessSet& operator=(const BindlessSet&) = delete;
    BindlessSet(BindlessSet&&) = delete;
    BindlessSet& operator=(BindlessSet&&) = delete;

    [[nodiscard]] VkDescriptorSetLayout layout() const { return layout_; }
    [[nodiscard]] VkDescriptorSet handle() const { return set_; }

    // Registers a buffer at 'index' in the storage-buffer array. Safe to call
    // while the set is bound, thanks to UPDATE_AFTER_BIND.
    void write_storage_buffer(std::uint32_t index, VkBuffer buffer, VkDeviceSize size) const;

    // Registers an image at 'index' in the sampled-image array. The image must be in
    // SHADER_READ_ONLY_OPTIMAL by the time a shader samples it. The descriptor records
    // that layout but nothing verifies it.
    void write_sampled_image(std::uint32_t index, VkImageView view, VkSampler sampler) const;

    // Registers the object-id attachment. Rewritten whenever the image is
    // recreated, which is every swapchain resize. Expects the image to be in
    // SHADER_READ_ONLY_OPTIMAL by the time a shader reads it.
    void write_object_id_image(VkImageView view) const;

    // Registers the HDR colour target. Rewritten on every swapchain resize, for
    // the same reason as the id attachment above.
    void write_hdr_color_image(VkImageView view) const;

    // Registers the shadow map with its comparison sampler. Written once: the
    // shadow map's size is a quality setting, not a function of the window, so
    // a resize leaves it alone.
    void write_shadow_map(VkImageView view, VkSampler sampler) const;

    // Registers the tonemapped image. Rewritten on every swapchain resize --
    // the view changes with the window, though the sampler beside it does not.
    void write_ldr_color_image(VkImageView view, VkSampler sampler) const;

private:
    const Device& device_;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
};

}  // namespace sage::gpu