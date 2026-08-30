#pragma once

#include <sage/core/math.hpp>
#include <sage/gpu/material.hpp>
#include <sage/gpu/scene.hpp>

#include <filesystem>
#include <vector>

namespace sage::gpu {

class GeometryRegistry;
class TextureRegistry;

// Everything a glTF load produces other than the nodes, which go straight into
// the SceneGraph rather than being returned.
struct LoadedScene {
    // Uploaded to the MaterialRegistry; a node's material_index indexes this.
    std::vector<Material> materials;
    // World-space, for framing a camera on an arbitrary model.
    glm::vec3 bounds_min{0.0F};
    glm::vec3 bounds_max{0.0F};
    // Everything from the file is parented under this one node, so a load can
    // later be moved, hidden or removed as a unit.
    NodeHandle root;
};

// Loads a glTF or GLB file, appending its hierarchy to `graph` rather than
// flattening it. Blocks until every mesh and texture is resident on the GPU.
[[nodiscard]] LoadedScene load_gltf(const std::filesystem::path& path, GeometryRegistry& registry,
                                    TextureRegistry& textures, SceneGraph& graph);

}  // namespace sage::gpu
