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
    // Normalised so "assets/x/../x/map.png" and "assets/x/map.png" share a slot.
    // weakly_canonical rather than canonical: it tolerates a path that does not
    // resolve, which the existence check below is about to report anyway.
    std::error_code error;
    std::filesystem::path key_path = std::filesystem::weakly_canonical(path, error);
    if (error) {
        key_path = path;
    }

    const auto key = std::make_pair(key_path, color_space);
    if (const auto cached = by_path_.find(key); cached != by_path_.end()) {
        return cached->second;
    }

    // Checked before constructing, because Texture aborts on a decode failure
    // and a missing map should degrade rather than take the model down.
    if (!std::filesystem::exists(key_path, error)) {
        SAGE_LOG_WARN("Texture not found, using fallback: {}", path.string());
        // Deliberately not cached: the fallback is not this texture, and a file
        // that appears later should be picked up on the next load.
        return k_fallback_slot;
    }

    const std::uint32_t slot = register_texture(
        std::make_unique<Texture>(allocator_, device_, uploader_, key_path, color_space));
    by_path_.emplace(key, slot);
    return slot;
}

void TextureRegistry::reset() {
    // Every scene slot is pointed back at the white fallback before its image
    // is destroyed. PARTIALLY_BOUND only excuses descriptors that were never
    // written; one left naming a dead view is dangling whether or not a shader
    // reaches it, and the fallback is still alive to name.
    for (std::uint32_t slot = k_reserved_slots; slot < next_slot_; ++slot) {
        bindless_set_.write_sampled_image(slot, textures_[k_fallback_slot]->view(),
                                          sampler_.handle());
    }

    textures_.resize(k_reserved_slots);
    next_slot_ = k_reserved_slots;
    by_path_.clear();
}

TextureRegistry::~TextureRegistry() = default;

}  // namespace sage::gpu