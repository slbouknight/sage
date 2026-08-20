#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/gltf_loader.hpp>
#include <sage/gpu/vertex.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace sage::gpu {

namespace {

fastgltf::Asset parse(const std::filesystem::path& path) {
    auto data = fastgltf::GltfDataBuffer::FromPath(path);
    if (data.error() != fastgltf::Error::None) {
        SAGE_LOG_ERROR("Could not read glTF file: {} ({})", path.string(),
                       fastgltf::getErrorMessage(data.error()));
    }
    SAGE_VERIFY(data.error() == fastgltf::Error::None, "Failed to read glTF file");

    fastgltf::Parser parser;
    // LoadExternalBuffers pulls in the .bin sidecar. Images are deliberately
    // not requested, so a glTF referencing textures we did not vendor still
    // loads cleanly.
    // GenerateMeshIndices synthesises indices for non-indexed primitives, so
    // the rest of the pipeline can assume an index buffer always exists.
    auto expected = parser.loadGltf(
        data.get(), path.parent_path(),
        fastgltf::Options::LoadExternalBuffers | fastgltf::Options::GenerateMeshIndices);
    if (expected.error() != fastgltf::Error::None) {
        SAGE_LOG_ERROR("Could not parse glTF: {} ({})", path.string(),
                       fastgltf::getErrorMessage(expected.error()));
    }
    SAGE_VERIFY(expected.error() == fastgltf::Error::None, "Failed to parse glTF");

    return std::move(expected.get());
}

}  // namespace

Scene load_gltf(const std::filesystem::path& path, GeometryRegistry& registry) {
    const fastgltf::Asset asset = parse(path);

    Scene scene;
    scene.bounds_min = glm::vec3(std::numeric_limits<float>::max());
    scene.bounds_max = glm::vec3(std::numeric_limits<float>::lowest());

    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;

    const std::size_t scene_index = asset.defaultScene.value_or(0);

    fastgltf::iterateSceneNodes(
        asset, scene_index, fastgltf::math::fmat4x4(),
        [&](const fastgltf::Node& node, const fastgltf::math::fmat4x4& world) {
            if (!node.meshIndex.has_value()) {
                return;
            }

            const glm::mat4 transform = glm::make_mat4(world.data());

            for (const fastgltf::Primitive& primitive : asset.meshes[*node.meshIndex].primitives) {
                if (primitive.type != fastgltf::PrimitiveType::Triangles) {
                    SAGE_LOG_WARN("Skipping non-triangle primitive");
                    continue;
                }

                const auto* position_attribute = primitive.findAttribute("POSITION");
                if (position_attribute == primitive.attributes.end()) {
                    SAGE_LOG_WARN("Skipping primitive with no POSITION attribute");
                    continue;
                }
                SAGE_VERIFY(primitive.indicesAccessor.has_value(),
                            "Primitive has no indices despite GenerateMeshIndices");

                const fastgltf::Accessor& position_accessor =
                    asset.accessors[position_attribute->accessorIndex];
                const std::size_t vertex_count = position_accessor.count;

                positions.resize(vertex_count);
                fastgltf::copyFromAccessor<glm::vec3>(asset, position_accessor, positions.data());

                // NORMAL is optional in glTF. Without it the mesh would be
                // unlit; a fixed up-facing normal at least keeps it visible.
                normals.assign(vertex_count, glm::vec3(0.0F, 1.0F, 0.0F));
                if (const auto* normal_attribute = primitive.findAttribute("NORMAL");
                    normal_attribute != primitive.attributes.end()) {
                    fastgltf::copyFromAccessor<glm::vec3>(
                        asset, asset.accessors[normal_attribute->accessorIndex], normals.data());
                } else {
                    SAGE_LOG_WARN("Primitive has no NORMAL attribute; shading will be flat");
                }

                // TEXCOORD_0 is optional in glTF. Zeroed UVs sample a single
                // texel rather than failing to load. Wrong, but visibly wrong.
                uvs.assign(vertex_count, glm::vec2(0.0F));
                if (const auto* uv_attribute = primitive.findAttribute("TEXCOORD_0");
                    uv_attribute != primitive.attributes.end()) {
                    fastgltf::copyFromAccessor<glm::vec2>(
                        asset, asset.accessors[uv_attribute->accessorIndex], uvs.data());
                } else {
                    SAGE_LOG_WARN("Primitive has no TEXCOORD_0; textures will not map");
                }

                vertices.resize(vertex_count);
                for (std::size_t i = 0; i < vertex_count; ++i) {
                    vertices[i].position = positions[i];
                    vertices[i].normal = normals[i];
                    vertices[i].uv = uvs[i];

                    const glm::vec3 world_position =
                        glm::vec3(transform * glm::vec4(positions[i], 1.0F));
                    scene.bounds_min = glm::min(scene.bounds_min, world_position);
                    scene.bounds_max = glm::max(scene.bounds_max, world_position);
                }

                const fastgltf::Accessor& index_accessor =
                    asset.accessors[*primitive.indicesAccessor];
                indices.resize(index_accessor.count);
                fastgltf::copyFromAccessor<std::uint32_t>(asset, index_accessor, indices.data());

                SceneNode scene_node;
                scene_node.transform = transform;
                scene_node.mesh =
                    registry.add_mesh(vertices.data(), sizeof(Vertex) * vertices.size(),
                                      indices.data(), static_cast<std::uint32_t>(indices.size()));
                scene.nodes.push_back(scene_node);
            }
        });

    SAGE_VERIFY(!scene.nodes.empty(), "glTF contained no drawable triangle primitives");
    SAGE_LOG_INFO(
        "Loaded {}: {} primitives, bounds ({:.2f}, {:.2f}, {:.2f}) to "
        "({:.2f}, {:.2f}, {:.2f})",
        path.filename().string(), scene.nodes.size(), scene.bounds_min.x, scene.bounds_min.y,
        scene.bounds_min.z, scene.bounds_max.x, scene.bounds_max.y, scene.bounds_max.z);

    return scene;
}

}  // namespace sage::gpu