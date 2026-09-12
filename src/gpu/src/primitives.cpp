#include <sage/core/assert.hpp>
#include <sage/gpu/primitives.hpp>

#include <cmath>
#include <numbers>

namespace sage::gpu {

namespace {

constexpr float k_pi = std::numbers::pi_v<float>;
constexpr float k_two_pi = 2.0F * k_pi;

void push_vertex(PrimitiveMesh& mesh, const glm::vec3& position, const glm::vec3& normal,
                 const glm::vec4& tangent, const glm::vec2& uv) {
    mesh.vertices.push_back(Vertex{position, normal, tangent, uv});
}

// Two triangles over four corners given in order around the quad. Split along
// a-c so both triangles keep the ring's winding.
void push_quad(PrimitiveMesh& mesh, std::uint32_t a, std::uint32_t b, std::uint32_t c,
               std::uint32_t d) {
    mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
}

// A flat disc in the XZ plane at `y`, facing `normal` (which must be +/-Y).
// Used for the caps of the cone and cylinder.
//
// The centre vertex is emitted first and the rim fanned around it. A fan is
// fine here: caps are small, and the alternative -- a triangulated grid -- buys
// nothing on a flat surface with a constant normal.
void push_cap(PrimitiveMesh& mesh, float y, float radius, bool facing_up) {
    const auto centre_index = static_cast<std::uint32_t>(mesh.vertices.size());
    const glm::vec3 normal = facing_up ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::vec3(0.0F, -1.0F, 0.0F);
    // +U runs along +X on both caps; the handedness sign is what differs, and
    // it is what keeps the bitangent pointing the same way round the disc.
    const glm::vec4 tangent{1.0F, 0.0F, 0.0F, facing_up ? 1.0F : -1.0F};

    push_vertex(mesh, glm::vec3(0.0F, y, 0.0F), normal, tangent, glm::vec2(0.5F, 0.5F));
    for (std::uint32_t i = 0; i <= k_primitive_segments; ++i) {
        const float angle =
            k_two_pi * (static_cast<float>(i) / static_cast<float>(k_primitive_segments));
        const float cos_a = std::cos(angle);
        const float sin_a = std::sin(angle);
        push_vertex(mesh, glm::vec3(radius * cos_a, y, radius * sin_a), normal, tangent,
                    glm::vec2((cos_a * 0.5F) + 0.5F, (sin_a * 0.5F) + 0.5F));
    }

    for (std::uint32_t i = 0; i < k_primitive_segments; ++i) {
        const std::uint32_t rim = centre_index + 1 + i;
        // Seen from +Y looking down, increasing angle runs clockwise, so an
        // up-facing cap reverses the rim pair to come out counter-clockwise
        // from outside. A down-facing cap is seen from the other side, where
        // the same order already reads counter-clockwise.
        if (facing_up) {
            mesh.indices.insert(mesh.indices.end(), {centre_index, rim + 1, rim});
        } else {
            mesh.indices.insert(mesh.indices.end(), {centre_index, rim, rim + 1});
        }
    }
}

PrimitiveMesh make_plane() {
    PrimitiveMesh mesh;

    const glm::vec3 normal{0.0F, 1.0F, 0.0F};
    const glm::vec4 tangent{1.0F, 0.0F, 0.0F, 1.0F};

    // Corners in counter-clockwise order seen from above, which is the side the
    // +Y normal faces.
    push_vertex(mesh, glm::vec3(-0.5F, 0.0F, 0.5F), normal, tangent, glm::vec2(0.0F, 1.0F));
    push_vertex(mesh, glm::vec3(0.5F, 0.0F, 0.5F), normal, tangent, glm::vec2(1.0F, 1.0F));
    push_vertex(mesh, glm::vec3(0.5F, 0.0F, -0.5F), normal, tangent, glm::vec2(1.0F, 0.0F));
    push_vertex(mesh, glm::vec3(-0.5F, 0.0F, -0.5F), normal, tangent, glm::vec2(0.0F, 0.0F));

    push_quad(mesh, 0, 1, 2, 3);
    return mesh;
}

PrimitiveMesh make_cube() {
    PrimitiveMesh mesh;

    // Twenty-four vertices, not eight. A cube corner belongs to three faces
    // with three different normals and three different tangent frames; sharing
    // it would average them and round the lighting off every edge.
    struct Face {
        glm::vec3 normal;
        glm::vec3 tangent;
    };
    constexpr std::array<Face, 6> k_faces{
        Face{{0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 0.0F}},    // +Z
        Face{{0.0F, 0.0F, -1.0F}, {-1.0F, 0.0F, 0.0F}},  // -Z
        Face{{1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, -1.0F}},   // +X
        Face{{-1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F}},   // -X
        Face{{0.0F, 1.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},    // +Y
        Face{{0.0F, -1.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},   // -Y
    };

    for (const Face& face : k_faces) {
        const glm::vec3 bitangent = glm::cross(face.normal, face.tangent);
        const glm::vec3 centre = face.normal * 0.5F;
        const auto base = static_cast<std::uint32_t>(mesh.vertices.size());

        // Walking tangent then bitangent around the face gives counter-
        // clockwise winding seen from along the normal, for every face, without
        // six hand-written vertex orders to get wrong.
        push_vertex(mesh, centre - (face.tangent * 0.5F) - (bitangent * 0.5F), face.normal,
                    glm::vec4(face.tangent, 1.0F), glm::vec2(0.0F, 0.0F));
        push_vertex(mesh, centre + (face.tangent * 0.5F) - (bitangent * 0.5F), face.normal,
                    glm::vec4(face.tangent, 1.0F), glm::vec2(1.0F, 0.0F));
        push_vertex(mesh, centre + (face.tangent * 0.5F) + (bitangent * 0.5F), face.normal,
                    glm::vec4(face.tangent, 1.0F), glm::vec2(1.0F, 1.0F));
        push_vertex(mesh, centre - (face.tangent * 0.5F) + (bitangent * 0.5F), face.normal,
                    glm::vec4(face.tangent, 1.0F), glm::vec2(0.0F, 1.0F));

        push_quad(mesh, base, base + 1, base + 2, base + 3);
    }
    return mesh;
}

PrimitiveMesh make_sphere() {
    PrimitiveMesh mesh;

    // The seam is duplicated: the vertex at u=0 and the one at u=1 are the same
    // point in space but not the same texture coordinate, and sharing them
    // would run the whole texture backwards across the last column.
    for (std::uint32_t ring = 0; ring <= k_primitive_rings; ++ring) {
        const float v = static_cast<float>(ring) / static_cast<float>(k_primitive_rings);
        const float theta = v * k_pi;
        const float sin_t = std::sin(theta);
        const float cos_t = std::cos(theta);

        for (std::uint32_t segment = 0; segment <= k_primitive_segments; ++segment) {
            const float u = static_cast<float>(segment) / static_cast<float>(k_primitive_segments);
            const float phi = u * k_two_pi;
            const float sin_p = std::sin(phi);
            const float cos_p = std::cos(phi);

            // Radius 0.5, so the sphere is a unit diameter like the cube is a
            // unit edge -- the two then scale comparably.
            const glm::vec3 normal{sin_t * cos_p, cos_t, sin_t * sin_p};
            // d(position)/du, which is the direction +U runs.
            const glm::vec3 tangent{-sin_p, 0.0F, cos_p};
            push_vertex(mesh, normal * 0.5F, normal, glm::vec4(tangent, 1.0F), glm::vec2(u, v));
        }
    }

    const std::uint32_t stride = k_primitive_segments + 1;
    for (std::uint32_t ring = 0; ring < k_primitive_rings; ++ring) {
        // The two rings at the poles are a whole row of vertices sitting on one
        // point -- distinct only in u, so the texture has somewhere to converge
        // to. The quad against them is really a triangle, and emitting it as a
        // quad puts a zero-area triangle in the buffer for every segment. They
        // would be culled rather than drawn wrong, which is exactly why this is
        // worth catching here instead of never noticing.
        const bool at_north_pole = ring == 0;
        const bool at_south_pole = ring == k_primitive_rings - 1;

        for (std::uint32_t segment = 0; segment < k_primitive_segments; ++segment) {
            const std::uint32_t a = (ring * stride) + segment;
            const std::uint32_t b = a + stride;

            if (!at_north_pole) {
                mesh.indices.insert(mesh.indices.end(), {a, a + 1, b + 1});
            }
            if (!at_south_pole) {
                mesh.indices.insert(mesh.indices.end(), {a, b + 1, b});
            }
        }
    }
    return mesh;
}

PrimitiveMesh make_cone() {
    PrimitiveMesh mesh;

    constexpr float k_radius = 0.5F;
    constexpr float k_height = 1.0F;
    constexpr float k_base_y = -0.5F;
    constexpr float k_apex_y = 0.5F;

    // The apex is duplicated per segment. One shared apex vertex would need a
    // single normal for a point where the surface has a different normal in
    // every direction, and would shade as a dimple.
    for (std::uint32_t i = 0; i < k_primitive_segments; ++i) {
        const float u0 = static_cast<float>(i) / static_cast<float>(k_primitive_segments);
        const float u1 = static_cast<float>(i + 1) / static_cast<float>(k_primitive_segments);
        const float a0 = u0 * k_two_pi;
        const float a1 = u1 * k_two_pi;
        // The apex normal is the average of its two neighbours' -- the best
        // available answer where the true normal is undefined.
        const float am = (a0 + a1) * 0.5F;

        // Perpendicular to the slant, not radial: a cone's side leans, so the
        // normal leans with it. Height and radius swap places here, which is
        // the part that looks wrong and is not.
        const auto side_normal = [](float angle) {
            return glm::normalize(
                glm::vec3(k_height * std::cos(angle), k_radius, k_height * std::sin(angle)));
        };
        const auto tangent_at = [](float angle) {
            return glm::vec4(-std::sin(angle), 0.0F, std::cos(angle), 1.0F);
        };

        const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
        push_vertex(mesh, glm::vec3(k_radius * std::cos(a0), k_base_y, k_radius * std::sin(a0)),
                    side_normal(a0), tangent_at(a0), glm::vec2(u0, 1.0F));
        push_vertex(mesh, glm::vec3(k_radius * std::cos(a1), k_base_y, k_radius * std::sin(a1)),
                    side_normal(a1), tangent_at(a1), glm::vec2(u1, 1.0F));
        push_vertex(mesh, glm::vec3(0.0F, k_apex_y, 0.0F), side_normal(am), tangent_at(am),
                    glm::vec2((u0 + u1) * 0.5F, 0.0F));

        // Reversed against increasing angle, for the same reason as the caps:
        // increasing angle runs clockwise when seen from outside.
        mesh.indices.insert(mesh.indices.end(), {base + 1, base, base + 2});
    }

    push_cap(mesh, k_base_y, k_radius, false);
    return mesh;
}

PrimitiveMesh make_cylinder() {
    PrimitiveMesh mesh;

    constexpr float k_radius = 0.5F;
    constexpr float k_top_y = 0.5F;
    constexpr float k_bottom_y = -0.5F;

    for (std::uint32_t i = 0; i <= k_primitive_segments; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(k_primitive_segments);
        const float angle = u * k_two_pi;
        const float cos_a = std::cos(angle);
        const float sin_a = std::sin(angle);

        const glm::vec3 normal{cos_a, 0.0F, sin_a};
        const glm::vec4 tangent{-sin_a, 0.0F, cos_a, 1.0F};
        push_vertex(mesh, glm::vec3(k_radius * cos_a, k_top_y, k_radius * sin_a), normal, tangent,
                    glm::vec2(u, 0.0F));
        push_vertex(mesh, glm::vec3(k_radius * cos_a, k_bottom_y, k_radius * sin_a), normal,
                    tangent, glm::vec2(u, 1.0F));
    }

    for (std::uint32_t i = 0; i < k_primitive_segments; ++i) {
        const std::uint32_t top = i * 2;
        const std::uint32_t bottom = top + 1;
        // Around the ring first, then down. The other order -- down, then
        // around -- reverses the winding and renders the whole side inside
        // out: back faces are culled, so the near surface disappears and the
        // far one shows through, lit from behind.
        push_quad(mesh, top, top + 2, bottom + 2, bottom);
    }

    push_cap(mesh, k_top_y, k_radius, true);
    push_cap(mesh, k_bottom_y, k_radius, false);
    return mesh;
}

}  // namespace

PrimitiveMesh make_primitive(PrimitiveKind kind) {
    switch (kind) {
        case PrimitiveKind::plane:
            return make_plane();
        case PrimitiveKind::cube:
            return make_cube();
        case PrimitiveKind::sphere:
            return make_sphere();
        case PrimitiveKind::cone:
            return make_cone();
        case PrimitiveKind::cylinder:
            return make_cylinder();
    }
    SAGE_VERIFY(false, "make_primitive: unhandled kind");
    return {};
}

const char* primitive_name(PrimitiveKind kind) {
    switch (kind) {
        case PrimitiveKind::plane:
            return "Plane";
        case PrimitiveKind::cube:
            return "Cube";
        case PrimitiveKind::sphere:
            return "Sphere";
        case PrimitiveKind::cone:
            return "Cone";
        case PrimitiveKind::cylinder:
            return "Cylinder";
    }
    return "Primitive";
}

}  // namespace sage::gpu
