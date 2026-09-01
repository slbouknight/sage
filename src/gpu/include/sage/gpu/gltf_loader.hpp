#pragma once

#include <sage/core/math.hpp>
#include <sage/gpu/scene.hpp>

#include <filesystem>
#include <optional>

namespace sage::gpu {

class GeometryRegistry;
class MaterialRegistry;
class TextureRegistry;

// Everything a glTF load produces other than the nodes, which go straight into
// the SceneGraph rather than being returned.
struct LoadedScene {
    // World-space, for framing a camera on an arbitrary model. Equal when the
    // file drew nothing.
    glm::vec3 bounds_min{0.0F};
    glm::vec3 bounds_max{0.0F};
    // Everything from the file is parented under this one node, so a load can
    // later be moved, hidden or removed as a unit.
    NodeHandle root;
    std::size_t mesh_count = 0;
};

// Loads a glTF or GLB file, appending its hierarchy to `graph` rather than
// flattening it, and its materials to `materials`. Blocks until every mesh and
// texture is resident on the GPU.
//
// Returns nothing when the file cannot be read or parsed, leaving every
// registry as it found them. The path comes from a file picker, so a bad choice
// has to be a message rather than an abort.
[[nodiscard]] std::optional<LoadedScene> load_gltf(const std::filesystem::path& path,
                                                   GeometryRegistry& registry,
                                                   TextureRegistry& textures,
                                                   MaterialRegistry& materials, SceneGraph& graph);

}  // namespace sage::gpu
