#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/gltf_loader.hpp>
#include <sage/gpu/scene.hpp>
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
#include <optional>
#include <string>
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
                            std::unordered_map<std::uint64_t, std::uint32_t>& cache,
                            TextureColorSpace color_space) {
    // Keyed on image and colour space together: one image decoded as sRGB and
    // as linear needs two slots, because the format is baked into the view.
    const std::uint64_t key = image_index * 2 + (color_space == TextureColorSpace::srgb ? 1 : 0);

    if (const auto cached = cache.find(key); cached != cache.end()) {
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
            slot = textures.add(base_directory / uri->uri.fspath(), color_space);
        }
    } else {
        // GLB-embedded and base64 images arrive as BufferView or Array. Both
        // hold bytes rather than a path, so they need stbi_load_from_memory --
        // additive, but not something M4 has.
        SAGE_LOG_WARN("Image {} is embedded rather than an external file; using fallback",
                      image_index);
    }

    cache.emplace(key, slot);
    return slot;
}

// Follows a glTF texture reference to the bindless slot its image landed in.
std::uint32_t resolve_texture(const fastgltf::Asset& asset, const fastgltf::TextureInfo& info,
                              const std::filesystem::path& base_directory,
                              TextureRegistry& textures,
                              std::unordered_map<std::uint64_t, std::uint32_t>& cache,
                              TextureColorSpace color_space) {
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

    return resolve_image(asset, *texture.imageIndex, base_directory, textures, cache, color_space);
}

// Reused across primitives so a scene with many meshes does not reallocate
// these on every one.
struct LoadScratch {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents;
    std::vector<glm::vec2> uvs;
};

struct LoadState {
    glm::vec3 bounds_min{std::numeric_limits<float>::max()};
    glm::vec3 bounds_max{std::numeric_limits<float>::lowest()};
    std::size_t mesh_count = 0;
};

// Uploads one primitive and returns how to draw it. Empty when the primitive is
// not drawable, which is a warning rather than a failure -- one bad primitive
// should not cost the whole file.
std::optional<GeometryRegistry::MeshView> upload_primitive(
    const fastgltf::Asset& asset, const fastgltf::Primitive& primitive, GeometryRegistry& registry,
    LoadScratch& scratch, const glm::mat4& world, LoadState& state) {
    if (primitive.type != fastgltf::PrimitiveType::Triangles) {
        SAGE_LOG_WARN("Skipping non-triangle primitive");
        return std::nullopt;
    }

    const auto* position_attribute = primitive.findAttribute("POSITION");
    if (position_attribute == primitive.attributes.end()) {
        SAGE_LOG_WARN("Skipping primitive with no POSITION attribute");
        return std::nullopt;
    }
    SAGE_VERIFY(primitive.indicesAccessor.has_value(),
                "Primitive has no indices despite GenerateMeshIndices");

    const fastgltf::Accessor& position_accessor =
        asset.accessors[position_attribute->accessorIndex];
    const std::size_t vertex_count = position_accessor.count;

    scratch.positions.resize(vertex_count);
    fastgltf::copyFromAccessor<glm::vec3>(asset, position_accessor, scratch.positions.data());

    // NORMAL is optional in glTF. Without it the mesh would be unlit; a fixed
    // up-facing normal at least keeps it visible.
    scratch.normals.assign(vertex_count, glm::vec3(0.0F, 1.0F, 0.0F));
    if (const auto* normal_attribute = primitive.findAttribute("NORMAL");
        normal_attribute != primitive.attributes.end()) {
        fastgltf::copyFromAccessor<glm::vec3>(
            asset, asset.accessors[normal_attribute->accessorIndex], scratch.normals.data());
    } else {
        SAGE_LOG_WARN("Primitive has no NORMAL attribute; shading will be flat");
    }

    // TANGENT is optional. The default is arbitrary but finite; a mesh without
    // tangents cannot orient a normal map, and deriving them from UVs is real
    // work no vendored asset needs.
    scratch.tangents.assign(vertex_count, glm::vec4(1.0F, 0.0F, 0.0F, 1.0F));
    if (const auto* tangent_attribute = primitive.findAttribute("TANGENT");
        tangent_attribute != primitive.attributes.end()) {
        fastgltf::copyFromAccessor<glm::vec4>(
            asset, asset.accessors[tangent_attribute->accessorIndex], scratch.tangents.data());
    } else {
        SAGE_LOG_WARN("Primitive has no TANGENT; normal mapping will be wrong");
    }

    // TEXCOORD_0 is optional. Zeroed UVs sample a single texel rather than
    // failing to load. Wrong, but visibly wrong.
    scratch.uvs.assign(vertex_count, glm::vec2(0.0F));
    if (const auto* uv_attribute = primitive.findAttribute("TEXCOORD_0");
        uv_attribute != primitive.attributes.end()) {
        fastgltf::copyFromAccessor<glm::vec2>(asset, asset.accessors[uv_attribute->accessorIndex],
                                              scratch.uvs.data());
    } else {
        SAGE_LOG_WARN("Primitive has no TEXCOORD_0; textures will not map");
    }

    scratch.vertices.resize(vertex_count);
    for (std::size_t i = 0; i < vertex_count; ++i) {
        scratch.vertices[i].position = scratch.positions[i];
        scratch.vertices[i].normal = scratch.normals[i];
        scratch.vertices[i].tangent = scratch.tangents[i];
        scratch.vertices[i].uv = scratch.uvs[i];

        // Bounds are world-space so a camera can frame the load; the graph has
        // not composed its transforms yet, so the walk carries the matrix down.
        const glm::vec3 world_position = glm::vec3(world * glm::vec4(scratch.positions[i], 1.0F));
        state.bounds_min = glm::min(state.bounds_min, world_position);
        state.bounds_max = glm::max(state.bounds_max, world_position);
    }

    const fastgltf::Accessor& index_accessor = asset.accessors[*primitive.indicesAccessor];
    scratch.indices.resize(index_accessor.count);
    fastgltf::copyFromAccessor<std::uint32_t>(asset, index_accessor, scratch.indices.data());

    ++state.mesh_count;
    return registry.add_mesh(scratch.vertices.data(), sizeof(Vertex) * scratch.vertices.size(),
                             scratch.indices.data(),
                             static_cast<std::uint32_t>(scratch.indices.size()));
}

// Adds one glTF node and everything beneath it. Depth-first, and the node is
// added before recursing into its children -- which is exactly the
// parents-precede-children invariant SceneGraph::add_node enforces.
void add_gltf_node(const fastgltf::Asset& asset, std::size_t node_index, NodeHandle parent,
                   const glm::mat4& parent_world, SceneGraph& graph, GeometryRegistry& registry,
                   LoadScratch& scratch, LoadState& state, std::uint32_t default_material_index) {
    const fastgltf::Node& node = asset.nodes[node_index];

    // getTransformMatrix against the default identity base yields this node's
    // own transform rather than a composed one. Composition is the graph's job
    // now, which is the entire point of keeping the hierarchy.
    const glm::mat4 local = glm::make_mat4(fastgltf::getTransformMatrix(node).data());
    const glm::mat4 world = parent_world * local;

    const std::string node_name =
        node.name.empty() ? "node " + std::to_string(node_index) : std::string(node.name.c_str());
    const NodeHandle handle = graph.add_node(parent, local, node_name);

    if (node.meshIndex.has_value()) {
        const fastgltf::Mesh& mesh = asset.meshes[*node.meshIndex];
        // A node carries at most one draw. A single-primitive mesh attaches
        // directly; several become children, so the hierarchy panel stays clean
        // for the common case instead of showing a wrapper per mesh.
        const bool attach_directly = mesh.primitives.size() == 1;

        std::size_t primitive_index = 0;
        for (const fastgltf::Primitive& primitive : mesh.primitives) {
            const std::optional<GeometryRegistry::MeshView> view =
                upload_primitive(asset, primitive, registry, scratch, world, state);

            if (view.has_value()) {
                const std::uint32_t material_index =
                    primitive.materialIndex.has_value()
                        ? static_cast<std::uint32_t>(*primitive.materialIndex)
                        : default_material_index;

                if (attach_directly) {
                    graph.set_mesh(handle, *view, material_index);
                } else {
                    const NodeHandle child = graph.add_node(
                        handle, glm::mat4(1.0F), node_name + "." + std::to_string(primitive_index));
                    graph.set_mesh(child, *view, material_index);
                }
            }
            ++primitive_index;
        }
    }

    for (const std::size_t child_index : node.children) {
        add_gltf_node(asset, child_index, handle, world, graph, registry, scratch, state,
                      default_material_index);
    }
}

}  // namespace

LoadedScene load_gltf(const std::filesystem::path& path, GeometryRegistry& registry,
                      TextureRegistry& textures, SceneGraph& graph) {
    const fastgltf::Asset asset = parse(path);

    LoadedScene loaded;
    // Index-for-index with the glTF's own material list, so a primitive's
    // materialIndex maps across untouched, plus one appended fallback carrying
    // glTF's prescribed defaults for primitives that name no material.
    loaded.materials.reserve(asset.materials.size() + 1);
    std::unordered_map<std::uint64_t, std::uint32_t> image_slots;
    for (const fastgltf::Material& source : asset.materials) {
        Material material;
        material.base_color_factor = glm::make_vec4(source.pbrData.baseColorFactor.data());
        material.metallic = source.pbrData.metallicFactor;
        material.roughness = source.pbrData.roughnessFactor;
        // Falls back to the 1x1 white texture, which is the multiplicative
        // identity against base_color_factor, so an untextured material needs
        // no special case in the shader.
        material.base_color_texture =
            source.pbrData.baseColorTexture.has_value()
                ? resolve_texture(asset, *source.pbrData.baseColorTexture, path.parent_path(),
                                  textures, image_slots, TextureColorSpace::srgb)
                : TextureRegistry::k_fallback_slot;

        // Linear: these carry measurements, not colour.
        material.metallic_roughness_texture =
            source.pbrData.metallicRoughnessTexture.has_value()
                ? resolve_texture(asset, *source.pbrData.metallicRoughnessTexture,
                                  path.parent_path(), textures, image_slots,
                                  TextureColorSpace::linear)
                : TextureRegistry::k_fallback_slot;

        material.normal_texture =
            source.normalTexture.has_value()
                ? resolve_texture(asset, *source.normalTexture, path.parent_path(), textures,
                                  image_slots, TextureColorSpace::linear)
                : TextureRegistry::k_flat_normal_slot;

        material.emissive_texture =
            source.emissiveTexture.has_value()
                ? resolve_texture(asset, *source.emissiveTexture, path.parent_path(), textures,
                                  image_slots, TextureColorSpace::srgb)
                : TextureRegistry::k_fallback_slot;

        material.emissive_factor = glm::make_vec3(source.emissiveFactor.data());
        material.normal_scale =
            source.normalTexture.has_value() ? source.normalTexture->scale : 1.0F;
        loaded.materials.push_back(material);
    }
    const auto default_material_index = static_cast<std::uint32_t>(loaded.materials.size());
    loaded.materials.emplace_back();

    // One node per load, so a file can be moved or removed as a unit and the
    // hierarchy panel has something to collapse.
    loaded.root = graph.add_node(NodeHandle{}, glm::mat4(1.0F), path.filename().string());

    LoadScratch scratch;
    LoadState state;
    const std::size_t scene_index = asset.defaultScene.value_or(0);
    for (const std::size_t node_index : asset.scenes[scene_index].nodeIndices) {
        add_gltf_node(asset, node_index, loaded.root, glm::mat4(1.0F), graph, registry, scratch,
                      state, default_material_index);
    }

    SAGE_VERIFY(state.mesh_count > 0, "glTF contained no drawable triangle primitives");

    // The graph owns composition from here; the walk's matrices only fed bounds.
    graph.update_transforms();

    loaded.bounds_min = state.bounds_min;
    loaded.bounds_max = state.bounds_max;

    SAGE_LOG_INFO(
        "Loaded {}: {} nodes, {} meshes, {} material(s) plus one default, bounds "
        "({:.2f}, {:.2f}, {:.2f}) to ({:.2f}, {:.2f}, {:.2f})",
        path.filename().string(), graph.size(), state.mesh_count, asset.materials.size(),
        loaded.bounds_min.x, loaded.bounds_min.y, loaded.bounds_min.z, loaded.bounds_max.x,
        loaded.bounds_max.y, loaded.bounds_max.z);

    return loaded;
}

}  // namespace sage::gpu
