#pragma once

#include <sage/core/handle.hpp>
#include <sage/core/math.hpp>
#include <sage/gpu/geometry_registry.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sage::gpu {

// Distinct tag so a node handle cannot be passed where a handle into some other
// pool is expected. See core::Handle.
struct SceneNodeTag;
using NodeHandle = core::Handle<SceneNodeTag>;

struct SceneNode {
    glm::mat4 local_transform{1.0F};
    // Derived: local composed with every ancestor. Only valid after
    // SceneGraph::update_transforms().
    glm::mat4 world_transform{1.0F};
    // Invalid for roots.
    NodeHandle parent;
    std::string name;

    // Renderable payload. Nodes without one are pure transforms -- glTF
    // hierarchy nodes now, and empties to parent things to once there is an
    // editor.
    GeometryRegistry::MeshView mesh;
    std::uint32_t material_index = 0;
    bool has_mesh = false;
};

// A mutable scene hierarchy, stored flat.
//
// Nodes live in one contiguous array ordered so that a parent always precedes
// its children. That invariant is the whole design: it lets update_transforms()
// be a single forward pass with no recursion, no visitor framework and no
// dirty-flag bookkeeping, because by the time a child is reached its parent's
// world transform is already final. add_node() enforces it rather than trusting
// callers to remember it -- a parent must already be in the array.
//
// This replaces M3's flattened Scene, which baked world transforms at load time
// and could not be edited (see ADR 0013).
class SceneGraph {
public:
    // `parent` may be a default-constructed handle, which makes a root.
    NodeHandle add_node(NodeHandle parent, const glm::mat4& local_transform, std::string name);

    void set_mesh(NodeHandle node, const GeometryRegistry::MeshView& mesh,
                  std::uint32_t material_index);
    void set_local_transform(NodeHandle node, const glm::mat4& local_transform);

    // Null when the handle is stale or was never valid.
    [[nodiscard]] const SceneNode* find(NodeHandle node) const;

    // Recomposes every world transform. Call after changing any local transform.
    void update_transforms();

    // Drops every node. Handles issued beforehand stop resolving.
    void clear();

    [[nodiscard]] std::span<const SceneNode> nodes() const { return nodes_; }
    [[nodiscard]] std::size_t size() const { return nodes_.size(); }
    [[nodiscard]] bool empty() const { return nodes_.empty(); }

private:
    [[nodiscard]] bool is_live(NodeHandle node) const;
    SceneNode* mutable_find(NodeHandle node);

    std::vector<SceneNode> nodes_;
    // Parallel to nodes_, but deliberately not shrunk by clear(): an index
    // reused after a clear must not resolve a handle issued before it.
    std::vector<NodeHandle::Generation> generations_;
};

}  // namespace sage::gpu
