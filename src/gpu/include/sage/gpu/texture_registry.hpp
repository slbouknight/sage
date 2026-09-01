#pragma once

#include <sage/gpu/texture.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace sage::gpu {

class Allocator;
class BindlessSet;
class Device;
class Sampler;
class Uploader;

// Owns every sampled image in the scene and hands out bindless slots.
//
// Slots are allocated here rather than in BindlessSet because this is the only
// thing that registers images; a counter on the set itself would be state with
// one writer and no other reader.
class TextureRegistry {
public:
    // Slot 0 is always a 1x1 opaque white texture. A material with no
    // base-colour map points at it and multiplies by its factor as usual, so
    // the shader needs no branch and never samples an unwritten descriptor --
    // which PARTIALLY_BOUND makes legal to leave empty but undefined to read.
    static constexpr std::uint32_t k_fallback_slot = 0;

    // Slot 1 is a 1x1 flat normal.  The white fallback above is the right
    // identity for base color, emissive, and metallic roughness (white times
    // a factor is that factor). But a normal map's identity is (0, 0, 1) and
    // white would decode to a normal tilted 5 deg off the surface.
    static constexpr std::uint32_t k_flat_normal_slot = 1;

    // The two fallbacks above are built once and survive reset(); everything
    // from here up is scene content.
    static constexpr std::uint32_t k_reserved_slots = 2;

    TextureRegistry(const Allocator& allocator, const Device& device, const Uploader& uploader,
                    const BindlessSet& bindless_set, const Sampler& sampler);
    ~TextureRegistry();

    TextureRegistry(const TextureRegistry&) = delete;
    TextureRegistry& operator=(const TextureRegistry&) = delete;
    TextureRegistry(TextureRegistry&&) = delete;
    TextureRegistry& operator=(TextureRegistry&&) = delete;

    // Decodes an image and registers it, returning its bindless slot. A missing
    // file yields k_fallback_slot and a warning rather than an abort: a glTF
    // may reference maps that were never vendored, and refusing to load the
    // whole model over one absent texture is the wrong trade for a viewer.
    // The same file in the same colour space is decoded once and shared across
    // loads, so adding a model twice costs one copy of its maps.
    [[nodiscard]] std::uint32_t add(const std::filesystem::path& path,
                                    TextureColorSpace color_space);

    // Destroys every scene texture, keeping the two fallbacks. Callers must
    // have waited for the device to be idle: these are live images.
    void reset();

    [[nodiscard]] std::uint32_t count() const { return next_slot_; }

private:
    std::uint32_t register_texture(std::unique_ptr<Texture> texture);

    const Allocator& allocator_;
    const Device& device_;
    const Uploader& uploader_;
    const BindlessSet& bindless_set_;
    const Sampler& sampler_;

    // unique_ptr because Texture is deliberately non-movable -- it owns raw
    // Vulkan handles -- so a vector of values could not reallocate.
    std::vector<std::unique_ptr<Texture>> textures_;
    std::uint32_t next_slot_ = 0;

    // Keyed on the file and the colour space together, because a format is
    // baked into the view: one image read as sRGB and as linear is two images.
    std::map<std::pair<std::filesystem::path, TextureColorSpace>, std::uint32_t> by_path_;
};

}  // namespace sage::gpu