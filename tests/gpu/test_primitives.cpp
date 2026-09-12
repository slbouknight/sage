#include <sage/gpu/primitives.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>

using Catch::Approx;
using sage::gpu::make_primitive;
using sage::gpu::primitive_name;
using sage::gpu::PrimitiveKind;
using sage::gpu::PrimitiveMesh;

namespace {

constexpr std::array<PrimitiveKind, 5> k_all_kinds{PrimitiveKind::plane, PrimitiveKind::cube,
                                                   PrimitiveKind::sphere, PrimitiveKind::cone,
                                                   PrimitiveKind::cylinder};

// A triangle's facing, taken from the order its vertices are wound in.
glm::vec3 geometric_normal(const PrimitiveMesh& mesh, std::size_t triangle) {
    const glm::vec3& a = mesh.vertices[mesh.indices[triangle * 3]].position;
    const glm::vec3& b = mesh.vertices[mesh.indices[(triangle * 3) + 1]].position;
    const glm::vec3& c = mesh.vertices[mesh.indices[(triangle * 3) + 2]].position;
    return glm::cross(b - a, c - a);
}

// The average of the three shading normals, which is where the surface is
// meant to face regardless of how the triangle happens to be wound.
glm::vec3 shading_normal(const PrimitiveMesh& mesh, std::size_t triangle) {
    return mesh.vertices[mesh.indices[triangle * 3]].normal +
           mesh.vertices[mesh.indices[(triangle * 3) + 1]].normal +
           mesh.vertices[mesh.indices[(triangle * 3) + 2]].normal;
}

}  // namespace

TEST_CASE("every primitive produces a non-degenerate indexed mesh", "[primitives]") {
    for (const PrimitiveKind kind : k_all_kinds) {
        const PrimitiveMesh mesh = make_primitive(kind);
        INFO(primitive_name(kind));

        CHECK(mesh.vertices.size() >= 3);
        CHECK(mesh.indices.size() >= 3);
        // Triangle list, so anything else is a generator that lost an index.
        CHECK(mesh.indices.size() % 3 == 0);

        for (const std::uint32_t index : mesh.indices) {
            REQUIRE(index < mesh.vertices.size());
        }
    }
}

// The one that matters most. The pipeline culls back faces with
// VK_FRONT_FACE_COUNTER_CLOCKWISE, so a primitive wound the wrong way renders
// inside-out: the near surface vanishes and the far one shows through, lit from
// behind. It still renders, which is exactly why it needs a test rather than
// a look.
TEST_CASE("triangles wind counter-clockwise seen from outside", "[primitives]") {
    for (const PrimitiveKind kind : k_all_kinds) {
        const PrimitiveMesh mesh = make_primitive(kind);
        INFO(primitive_name(kind));

        const std::size_t triangles = mesh.indices.size() / 3;
        for (std::size_t triangle = 0; triangle < triangles; ++triangle) {
            const glm::vec3 wound = geometric_normal(mesh, triangle);
            const glm::vec3 shaded = shading_normal(mesh, triangle);

            // A cone's apex triangles are thin, so compare directions rather
            // than requiring any particular magnitude.
            INFO("triangle " << triangle);
            CHECK(glm::dot(wound, shaded) > 0.0F);
        }
    }
}

TEST_CASE("normals are unit length", "[primitives]") {
    for (const PrimitiveKind kind : k_all_kinds) {
        const PrimitiveMesh mesh = make_primitive(kind);
        INFO(primitive_name(kind));
        for (const sage::gpu::Vertex& vertex : mesh.vertices) {
            CHECK(glm::length(vertex.normal) == Approx(1.0F).margin(1e-4F));
        }
    }
}

// apply_normal_map re-orthogonalises the tangent against the normal, so a
// tangent that is merely close is fine -- but one parallel to the normal
// collapses under Gram-Schmidt and leaves a degenerate basis.
TEST_CASE("tangents are usable as a basis", "[primitives]") {
    for (const PrimitiveKind kind : k_all_kinds) {
        const PrimitiveMesh mesh = make_primitive(kind);
        INFO(primitive_name(kind));
        for (const sage::gpu::Vertex& vertex : mesh.vertices) {
            const glm::vec3 tangent{vertex.tangent};
            CHECK(glm::length(tangent) == Approx(1.0F).margin(1e-4F));
            // glTF's handedness sign, not a free component.
            CHECK(std::abs(vertex.tangent.w) == Approx(1.0F).margin(1e-6F));
            CHECK(std::abs(glm::dot(glm::normalize(tangent), vertex.normal)) < 0.99F);
        }
    }
}

TEST_CASE("uvs stay inside the unit square", "[primitives]") {
    for (const PrimitiveKind kind : k_all_kinds) {
        const PrimitiveMesh mesh = make_primitive(kind);
        INFO(primitive_name(kind));
        for (const sage::gpu::Vertex& vertex : mesh.vertices) {
            CHECK(vertex.uv.x >= -1e-5F);
            CHECK(vertex.uv.x <= 1.0F + 1e-5F);
            CHECK(vertex.uv.y >= -1e-5F);
            CHECK(vertex.uv.y <= 1.0F + 1e-5F);
        }
    }
}

// Unit size around the origin is what lets the caller scale a primitive into
// any scene with one factor, and lift it onto the ground with one offset.
TEST_CASE("primitives are unit sized and centred", "[primitives]") {
    for (const PrimitiveKind kind : k_all_kinds) {
        const PrimitiveMesh mesh = make_primitive(kind);
        INFO(primitive_name(kind));

        glm::vec3 low{std::numeric_limits<float>::max()};
        glm::vec3 high{std::numeric_limits<float>::lowest()};
        for (const sage::gpu::Vertex& vertex : mesh.vertices) {
            low = glm::min(low, vertex.position);
            high = glm::max(high, vertex.position);
        }

        CHECK(low.x == Approx(-0.5F).margin(1e-4F));
        CHECK(high.x == Approx(0.5F).margin(1e-4F));
        CHECK(low.z == Approx(-0.5F).margin(1e-4F));
        CHECK(high.z == Approx(0.5F).margin(1e-4F));

        if (kind == PrimitiveKind::plane) {
            // Flat by definition, and the one primitive with no thickness.
            CHECK(low.y == Approx(0.0F).margin(1e-4F));
            CHECK(high.y == Approx(0.0F).margin(1e-4F));
        } else {
            CHECK(low.y == Approx(-0.5F).margin(1e-4F));
            CHECK(high.y == Approx(0.5F).margin(1e-4F));
        }
    }
}

TEST_CASE("the plane faces up", "[primitives]") {
    const PrimitiveMesh mesh = make_primitive(PrimitiveKind::plane);
    for (const sage::gpu::Vertex& vertex : mesh.vertices) {
        CHECK(vertex.normal.y == Approx(1.0F).margin(1e-5F));
    }
    // A ground plane lit from above and wound the wrong way would be culled
    // away entirely, so check the winding against +Y directly rather than
    // trusting the generic test to have caught it.
    CHECK(geometric_normal(mesh, 0).y > 0.0F);
    CHECK(geometric_normal(mesh, 1).y > 0.0F);
}

TEST_CASE("the cube has one flat normal per face", "[primitives]") {
    const PrimitiveMesh mesh = make_primitive(PrimitiveKind::cube);
    // Six faces of four corners: sharing the eight geometric corners would
    // average three normals together and round off every edge.
    CHECK(mesh.vertices.size() == 24);
    CHECK(mesh.indices.size() == 36);

    for (const sage::gpu::Vertex& vertex : mesh.vertices) {
        // Axis-aligned, so exactly one component is +/-1 and the rest are zero.
        const glm::vec3 n = vertex.normal;
        const float sum = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
        CHECK(sum == Approx(1.0F).margin(1e-5F));
    }
}

TEST_CASE("the sphere's normals point away from its centre", "[primitives]") {
    const PrimitiveMesh mesh = make_primitive(PrimitiveKind::sphere);
    for (const sage::gpu::Vertex& vertex : mesh.vertices) {
        // Radius 0.5 everywhere, and the normal is the outward radial.
        CHECK(glm::length(vertex.position) == Approx(0.5F).margin(1e-4F));
        if (glm::length(vertex.position) > 1e-4F) {
            CHECK(glm::dot(glm::normalize(vertex.position), vertex.normal) ==
                  Approx(1.0F).margin(1e-4F));
        }
    }
}

// The one piece of geometry here that is genuinely easy to get wrong: a cone's
// side normal is not radial. It leans by the slope, and height and radius swap
// places in the expression.
TEST_CASE("the cone's side normals lean with the slope", "[primitives]") {
    const PrimitiveMesh mesh = make_primitive(PrimitiveKind::cone);

    bool saw_side = false;
    for (const sage::gpu::Vertex& vertex : mesh.vertices) {
        if (vertex.normal.y < -0.99F) {
            continue;  // base cap
        }
        saw_side = true;
        // radius 0.5, height 1: normalize(h*cos, r, h*sin) has y = r / len,
        // where len = sqrt(h^2 + r^2) = sqrt(1.25).
        CHECK(vertex.normal.y == Approx(0.5F / std::sqrt(1.25F)).margin(1e-4F));
    }
    CHECK(saw_side);
}

TEST_CASE("the cylinder is closed at both ends", "[primitives]") {
    const PrimitiveMesh mesh = make_primitive(PrimitiveKind::cylinder);

    bool saw_up = false;
    bool saw_down = false;
    bool saw_side = false;
    for (const sage::gpu::Vertex& vertex : mesh.vertices) {
        if (vertex.normal.y > 0.99F) {
            saw_up = true;
        } else if (vertex.normal.y < -0.99F) {
            saw_down = true;
        } else if (std::abs(vertex.normal.y) < 1e-4F) {
            saw_side = true;
        }
    }
    CHECK(saw_up);
    CHECK(saw_down);
    CHECK(saw_side);
}
