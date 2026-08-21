#pragma once

#include <sage/core/math.hpp>
#include <sage/gpu/geometry_registry.hpp>
#include <sage/gpu/material.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

namespace sage::gpu {

class TextureRegistry;

// One drawable primitive plus where it sits in the scene.

struct SceneNode {
    GeometryRegistry::MeshView mesh;
    glm::mat4 transform{1.0F};
    // Index into Scene::materials, not into glTF's own material list
    std::uint32_t material_index = 0;
};

// A loaded scene, flattened: the node hierarchy is already resolved to world
// transforms, so drawing is a flat loop with no tree walk per frame.
struct Scene {
    std::vector<SceneNode> nodes;
    // World-space bounds, for framing a camera on an arbitrary model.
    glm::vec3 bounds_min{0.0F};
    glm::vec3 bounds_max{0.0F};
    // The scene's material table, uploaded verbatim to the MaterialRegistry.
    std::vector<Material> materials;
};

// Loads geometry from a glTF or GLB file into the registry. Materials,
// textures, animations, and cameras are ignored for now. Blocks until every mesh
// is resident on the GPU.
[[nodiscard]] Scene load_gltf(const std::filesystem::path& path, GeometryRegistry& registry,
                              TextureRegistry& textures);

}  // namespace sage::gpu