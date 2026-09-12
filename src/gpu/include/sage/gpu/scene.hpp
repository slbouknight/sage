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

    // False for a node that has been deleted. It stays in the array rather than
    // being erased; see remove_subtree for why.
    bool alive = true;
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

    // Deletes a node and everything beneath it.
    //
    // Tombstoned, not erased. Handles carry an index into this array, so
    // compacting it would silently repoint every handle past the hole --
    // deleting node 3 would leave a handle to node 5 naming node 6. Marking
    // instead keeps every surviving handle correct, keeps parents ahead of
    // their children without a re-sort, and keeps object-id picking valid,
    // since an id is an index and indices no longer move.
    //
    // The cost is that the array only grows, and that the deleted node's
    // geometry stays resident -- the registries bump-allocate and free nothing
    // below a full reset (see ADR 0011). That same property is what makes
    // restore_subtree free, and undo with it.
    void remove_subtree(NodeHandle node);
    // Puts one back, for undo. Restores exactly the nodes remove_subtree took,
    // which is why it works from the same handle.
    void restore_subtree(NodeHandle node);

    // Whether a handle names a node that still exists. find() returns null for
    // a deleted one; this separates "deleted" from "never valid", which undo
    // needs and a caller walking the scene does not.
    [[nodiscard]] bool is_deleted(NodeHandle node) const;

    // Recomposes every world transform. Call after changing any local transform.
    void update_transforms();

    // Drops every node. Handles issued beforehand stop resolving.
    void clear();

    // Every slot, including tombstoned ones. Callers that walk this must skip
    // nodes whose `alive` is false; there is no filtered view because the index
    // is the identity -- an id, a selection flag and a parent all name a slot,
    // and a compacted view would renumber them.
    [[nodiscard]] std::span<const SceneNode> nodes() const { return nodes_; }
    // Slots, not live nodes. live_size() is what a read-out should show.
    [[nodiscard]] std::size_t size() const { return nodes_.size(); }
    [[nodiscard]] std::size_t live_size() const;
    [[nodiscard]] bool empty() const { return nodes_.empty(); }

private:
    // Whether the handle resolves to a slot at all -- index in range and
    // generation current. Says nothing about whether that slot is alive.
    [[nodiscard]] bool is_live(NodeHandle node) const;
    SceneNode* mutable_find(NodeHandle node);
    // Marks a node and its descendants, for remove and restore alike.
    void set_subtree_alive(NodeHandle node, bool alive);

    std::vector<SceneNode> nodes_;
    // Parallel to nodes_, but deliberately not shrunk by clear(): an index
    // reused after a clear must not resolve a handle issued before it.
    std::vector<NodeHandle::Generation> generations_;
};

}  // namespace sage::gpu
