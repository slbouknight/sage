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
    glm::vec2 uv;
};
static_assert(sizeof(Vertex) == 32);
static_assert(offsetof(Vertex, position) == 0);
static_assert(offsetof(Vertex, normal) == 12);
static_assert(offsetof(Vertex, uv) == 24);

}  // namespace sage::gpu