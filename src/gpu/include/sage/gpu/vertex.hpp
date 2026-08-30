#pragma once

#include <sage/core/math.hpp>

#include <cstddef>

namespace sage::gpu {

// Must match Vertex in shaders/mesh.slang. Slang's "natural" layout for
// BDA-accessed structs is C-like packing -- no std430 vec3 padding -- which is
// exactly what glm gives. Verified against compiled SPIR-V:
// ArrayStride 24, offsets 0 and 12.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    // glTF's TANGENT: xyz is the tangent, w is a handedness sign of +/- 1 that
    // orients the bitangent. It is not a padding component.
    glm::vec4 tangent;
    glm::vec2 uv;
};
static_assert(sizeof(Vertex) == 48);
static_assert(offsetof(Vertex, position) == 0);
static_assert(offsetof(Vertex, normal) == 12);
static_assert(offsetof(Vertex, tangent) == 24);
static_assert(offsetof(Vertex, uv));

}  // namespace sage::gpu