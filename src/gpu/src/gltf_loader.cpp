#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/gltf_loader.hpp>
#include <sage/gpu/texture_registry.hpp>
#include <sage/gpu/vertex.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <variant>
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

// Maps a glTF image to a bindless slot, decoding it at most once no matter how
// many textures reference it -- glTF routinely points several textures at one
// image, and each decode is a full staging upload plus a mip chain.
std::uint32_t resolve_image(const fastgltf::Asset& asset, std::size_t image_index,
                            const std::filesystem::path& base_directory, TextureRegistry& textures,
                            std::unordered_map<std::size_t, std::uint32_t>& cache) {
    if (const auto cached = cache.find(image_index); cached != cache.end()) {
        return cached->second;
    }

    std::uint32_t slot = TextureRegistry::k_fallback_slot;
    const fastgltf::DataSource& source = asset.images[image_index].data;

    if (const auto* uri = std::get_if<fastgltf::sources::URI>(&source)) {
        if (!uri->uri.isLocalPath()) {
            SAGE_LOG_WARN("Image {} is not a local file ({}); using fallback", image_index,
                          uri->uri.c_str());
        } else if (uri->fileByteOffset != 0) {
            // An offset means the image sits inside a larger file, which needs
            // a decode-from-memory path rather than a plain open.
            SAGE_LOG_WARN("Image {} has a non-zero file offset; using fallback", image_index);
        } else {
            slot = textures.add(base_directory / uri->uri.fspath());
        }
    } else {
        // GLB-embedded and base64 images arrive as BufferView or Array. Both
        // hold bytes rather than a path, so they need stbi_load_from_memory --
        // additive, but not something M4 has.
        SAGE_LOG_WARN("Image {} is embedded rather than an external file; using fallback",
                      image_index);
    }

    cache.emplace(image_index, slot);
    return slot;
}

// Follows a glTF texture reference to the bindless slot its image landed in.
std::uint32_t resolve_texture(const fastgltf::Asset& asset, const fastgltf::TextureInfo& info,
                              const std::filesystem::path& base_directory,
                              TextureRegistry& textures,
                              std::unordered_map<std::size_t, std::uint32_t>& cache) {
    if (info.texCoordIndex != 0) {
        // Only TEXCOORD_0 is read into Vertex, so a second UV set would sample
        // with the wrong coordinates rather than fail loudly.
        SAGE_LOG_WARN("Texture uses TEXCOORD_{}; only TEXCOORD_0 is loaded", info.texCoordIndex);
    }

    const fastgltf::Texture& texture = asset.textures[info.textureIndex];
    if (!texture.imageIndex.has_value()) {
        // Extensions such as KHR_texture_basisu carry the image elsewhere.
        SAGE_LOG_WARN("Texture {} has no plain image source; using fallback", info.textureIndex);
        return TextureRegistry::k_fallback_slot;
    }

    return resolve_image(asset, *texture.imageIndex, base_directory, textures, cache);
}

}  // namespace

Scene load_gltf(const std::filesystem::path& path, GeometryRegistry& registry,
                TextureRegistry& textures) {
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

    // Index-for-index with the glTF's own material list, so a primitive's
    // materialIndex maps across untouched, plus one appended fallback carrying
    // glTF's prescribed defaults for primitives that name no material.
    scene.materials.reserve(asset.materials.size() + 1);
    std::unordered_map<std::size_t, std::uint32_t> image_slots;
    for (const fastgltf::Material& source : asset.materials) {
        Material material;
        material.base_color_factor = glm::make_vec4(source.pbrData.baseColorFactor.data());
        material.metallic = source.pbrData.metallicFactor;
        material.roughness = source.pbrData.roughnessFactor;
        // Falls back to the 1x1 white texture
        material.base_color_texture =
            source.pbrData.baseColorTexture.has_value()
                ? resolve_texture(asset, *source.pbrData.baseColorTexture, path.parent_path(),
                                  textures, image_slots)
                : TextureRegistry::k_fallback_slot;
        scene.materials.push_back(material);
    }
    const auto default_material_index = static_cast<std::uint32_t>(scene.materials.size());
    scene.materials.emplace_back();

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
                scene_node.material_index =
                    primitive.materialIndex.has_value()
                        ? static_cast<std::uint32_t>(*primitive.materialIndex)
                        : default_material_index;
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
    SAGE_LOG_INFO("Materials: {} from glTF, plus one default", asset.materials.size());

    return scene;
}

}  // namespace sage::gpu