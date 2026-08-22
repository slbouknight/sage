#pragma once

#include <sage/core/math.hpp>

#include <cstddef>
#include <cstdint>

namespace sage::gpu {
// Must match Material in shaders/mesh.slang. Read out of a StructuredBuffer,
// so this is std430: the vec4 forces 16-byte alignment and the trailing three
// scalars are absorbed into the struct's own padding, which is why metallic and
// roughness are free to carry even though nothing reads them yet.
//
// glm's vec4 has alignment 4, not 16, so the alginas is what
// makes the C++ offsets agree with the shader's. Verified against compiled
// SPIR-V: ArrayStride 32, member offsets 0, 16, 20 and 24.
struct Material {
    alignas(16) glm::vec4 base_color_factor{1.0F, 1.0F, 1.0F, 1.0F};
    glm::vec3 emissive_factor{0.0F, 0.0F, 0.0F};
    float metallic = 1.0F;
    float roughness = 1.0F;
    // Indices into the bindless sampled-image array, not glTF texture indices.
    std::uint32_t base_color_texture = 0;
    std::uint32_t normal_texture = 0;
    std::uint32_t metallic_roughness_texture = 0;
    std::uint32_t emissive_texture = 0;
    // glTF's normalTexture.scale; used in 3b.
    float normal_scale = 1.0F;
    // Not decoration: without these the struct strides at 52 under scalar
    // layout and 64 under std430. It is read through a descriptor today, so 64
    // is what counts -- but a deferred pass or a hit shader would reach it by
    // device address, and would then walk at 52. Verified: padded to 64 the two
    // layouts agree, which is the same property Light was given.
    float pad0_ = 0.0F;
    float pad1_ = 0.0F;
};
static_assert(sizeof(Material) == 64);
static_assert(offsetof(Material, base_color_factor) == 0);
static_assert(offsetof(Material, emissive_factor) == 16);
static_assert(offsetof(Material, metallic) == 28);
static_assert(offsetof(Material, roughness) == 32);
static_assert(offsetof(Material, base_color_texture) == 36);
static_assert(offsetof(Material, normal_texture) == 40);
static_assert(offsetof(Material, metallic_roughness_texture) == 44);
static_assert(offsetof(Material, emissive_texture) == 48);
static_assert(offsetof(Material, normal_scale) == 52);

}  // namespace sage::gpu