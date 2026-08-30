#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/bindless_set.hpp>
#include <sage/gpu/sampler.hpp>
#include <sage/gpu/texture.hpp>
#include <sage/gpu/texture_registry.hpp>

#include <array>
#include <utility>

namespace sage::gpu {

TextureRegistry::TextureRegistry(const Allocator& allocator, const Device& device,
                                 const Uploader& uploader, const BindlessSet& bindless_set,
                                 const Sampler& sampler)
    : allocator_(allocator),
      device_(device),
      uploader_(uploader),
      bindless_set_(bindless_set),
      sampler_(sampler) {
    // White rather than magenta: the fallback is multiplied by the material's
    // base_color_factor, so white is the identity and a factor-only material
    // renders correctly instead of announcing itself as an error.
    constexpr std::array<std::uint8_t, 4> k_white{255, 255, 255, 255};
    const std::uint32_t slot = register_texture(std::make_unique<Texture>(
        allocator_, device_, uploader_, k_white.data(), VkExtent2D{1, 1}, TextureColorSpace::srgb));
    SAGE_VERIFY(slot == k_fallback_slot, "Fallback texture must land in slot 0");

    // Linear, not sRGB: an sRGB decode turns 128 into 0.216 rather than 0.502,
    // which would bias every fallback normal instead of leaving it flat.
    constexpr std::array<std::uint8_t, 4> k_flat_normal{128, 128, 255, 255};
    const std::uint32_t normal_slot = register_texture(
        std::make_unique<Texture>(allocator_, device_, uploader_, k_flat_normal.data(),
                                  VkExtent2D{1, 1}, TextureColorSpace::linear));
    SAGE_VERIFY(normal_slot == k_flat_normal_slot, "Flat-normal texture must land in slot 1");
    SAGE_LOG_INFO("Texture registry: white fallback at slot {}, flat normal at slot {}", slot,
                  normal_slot);
}

std::uint32_t TextureRegistry::register_texture(std::unique_ptr<Texture> texture) {
    SAGE_VERIFY(next_slot_ < BindlessSet::k_max_sampled_images,
                "TextureRegistry: bindless sampled-image array is full");

    const std::uint32_t slot = next_slot_;
    bindless_set_.write_sampled_image(slot, texture->view(), sampler_.handle());
    textures_.push_back(std::move(texture));
    ++next_slot_;
    return slot;
}

std::uint32_t TextureRegistry::add(const std::filesystem::path& path,
                                   TextureColorSpace color_space) {
    // Checked before constructing, because Texture aborts on a decode failure
    // and a missing map should degrade rather than take the model down.
    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
        SAGE_LOG_WARN("Texture not found, using fallback: {}", path.string());
        return k_fallback_slot;
    }

    return register_texture(
        std::make_unique<Texture>(allocator_, device_, uploader_, path, color_space));
}

TextureRegistry::~TextureRegistry() = default;

}  // namespace sage::gpu