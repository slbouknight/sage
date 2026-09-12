#pragma once

#include <sage/gpu/vertex.hpp>

#include <cstdint>
#include <vector>

namespace sage::gpu {

// Procedural meshes, generated on the CPU in the same Vertex layout glTF loads
// into. They exist so a scene can be composed rather than only assembled from
// files -- above all a ground plane, without which a single loaded model has
// nothing for its shadow to fall on.
//
// Everything here is pure: no device, no registry, no allocation beyond the
// vectors returned. That is deliberate, because winding and tangent frames are
// exactly the sort of thing that is wrong in a way you only notice as a subtly
// mis-lit surface, and pure functions can be tested.
enum class PrimitiveKind : std::uint8_t {
    plane,
    cube,
    sphere,
    cone,
    cylinder,
};

struct PrimitiveMesh {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

// All of these are generated at unit size around the origin, and are scaled
// into a scene by the caller: a fixed size cannot suit both a chess piece and
// a street lamp, and the two differ by a factor of 36 among the sample models
// alone.
//
// Conventions, held by every generator and checked by the tests:
//   - Triangles wind counter-clockwise seen from outside, matching the
//     pipeline's VK_FRONT_FACE_COUNTER_CLOCKWISE with back-face culling on.
//     Get this backwards and the mesh renders inside-out but still renders.
//   - Normals are unit length and point out of the surface.
//   - Tangents follow glTF: xyz runs along +U, w is the handedness sign that
//     orients the bitangent as cross(normal, tangent) * w.
//   - UVs lie in [0, 1].
[[nodiscard]] PrimitiveMesh make_primitive(PrimitiveKind kind);

// For menu labels and logging.
[[nodiscard]] const char* primitive_name(PrimitiveKind kind);

// Rings of latitude and segments of longitude on the curved primitives. 32 is
// smooth enough that a sphere reads as round at screenshot resolution without
// the silhouette faceting, and small enough that the whole mesh is a few
// thousand triangles.
inline constexpr std::uint32_t k_primitive_segments = 32;
inline constexpr std::uint32_t k_primitive_rings = 16;

}  // namespace sage::gpu
