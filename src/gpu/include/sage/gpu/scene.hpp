#pragma once

#include <sage/core/handle.hpp>
#include <sage/core/math.hpp>
#include <sage/gpu/geometry_registry.hpp>
#include <sage/gpu/light.hpp>

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

// What a light node authors. Deliberately not gpu::Light: that struct is the
// GPU's layout, and half of it -- position and direction -- is derived from the
// node's transform rather than edited. Keeping the authored fields separate
// means there is exactly one place to change a light, and no field that looks
// editable but is overwritten every frame.
struct SceneLight {
    LightType type = LightType::directional;
    glm::vec3 color{1.0F, 1.0F, 1.0F};
    float intensity = 4.0F;
    // Point lights only. Distance at which influence reaches zero; 0 leaves a
    // pure inverse square, which never quite does.
    float range = 10.0F;
};

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

    // Lighting payload. A light is a node like anything else, so it can be
    // placed, parented, moved with the gizmo and listed in the hierarchy --
    // which is what makes it authorable at all.
    //
    // Position and direction are deliberately absent: they are the node's world
    // transform, derived when the frame is written. Storing them here as well
    // would be two sources of truth, and the gizmo would move only one.
    SceneLight light;
    bool has_light = false;

    // Editor furniture rather than scene content: the small mesh that makes a
    // light visible and clickable. Drawn in the viewport, but kept out of the
    // shadow pass -- an icon marking a light must not cast a shadow of its own
    // -- and out of captures, which are meant to be the render, not the tool.
    bool editor_only = false;
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
    void set_light(NodeHandle node, const SceneLight& light);
    // Marks a node as editor furniture: drawn in the viewport so it can be
    // seen and clicked, but excluded from the shadow pass and from captures.
    void set_editor_only(NodeHandle node, bool editor_only);

    // Null when the handle is stale or was never valid.
    [[nodiscard]] const SceneNode* find(NodeHandle node) const;

    // The handle naming the node at `index`, or a default handle when the
    // index is out of range. This is the inverse of nodes()[i] and exists for
    // object-ID picking: the GPU can only hand back an index, and a raw index
    // is worthless a scene-clear later. Converting it here, while the index is
    // still fresh, gives a handle whose generation catches exactly that.
    [[nodiscard]] NodeHandle handle_at(std::size_t index) const;

    // The topmost ancestor of `node`, or `node` itself when it is already a
    // root. Clicking a mesh in the viewport selects this rather than the mesh,
    // which is what makes a loaded file behave as one object to drag around.
    // A stale or invalid handle returns a default handle.
    [[nodiscard]] NodeHandle root_of(NodeHandle node) const;

    // Writes 1 for `node` and every node beneath it, 0 elsewhere. One forward
    // pass, for the same reason update_transforms() is one: a parent's answer
    // is always settled before its children are reached. `flags` is resized to
    // size(). An invalid handle clears every flag.
    void mark_subtree(NodeHandle node, std::vector<std::uint32_t>& flags) const;

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
