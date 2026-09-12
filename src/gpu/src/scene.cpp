#include <sage/core/assert.hpp>
#include <sage/gpu/scene.hpp>

#include <utility>

namespace sage::gpu {

namespace {
// core::Handle treats generation 0 as invalid, so live nodes start at 1.
constexpr NodeHandle::Generation k_first_generation = 1;
}  // namespace

NodeHandle SceneGraph::add_node(NodeHandle parent, const glm::mat4& local_transform,
                                std::string name) {
    // The parents-precede-children invariant is enforced, not merely documented:
    // a parent must already occupy a lower index than the child about to be
    // appended, which is exactly what requiring it to resolve guarantees.
    SAGE_VERIFY(!parent.valid() || is_live(parent),
                "SceneGraph: parent must be added before its children");

    const auto index = static_cast<NodeHandle::Index>(nodes_.size());
    if (index == generations_.size()) {
        generations_.push_back(k_first_generation);
    }

    SceneNode node;
    node.local_transform = local_transform;
    node.parent = parent;
    node.name = std::move(name);
    nodes_.push_back(std::move(node));

    return NodeHandle{index, generations_[index]};
}

bool SceneGraph::is_live(NodeHandle node) const {
    return node.valid() && node.index() < nodes_.size() &&
           generations_[node.index()] == node.generation();
}

const SceneNode* SceneGraph::find(NodeHandle node) const {
    return is_live(node) ? &nodes_[node.index()] : nullptr;
}

NodeHandle SceneGraph::handle_at(std::size_t index) const {
    if (index >= nodes_.size()) {
        return {};
    }
    return NodeHandle{static_cast<NodeHandle::Index>(index), generations_[index]};
}

NodeHandle SceneGraph::root_of(NodeHandle node) const {
    if (!is_live(node)) {
        return {};
    }

    NodeHandle current = node;
    // Terminates because a parent always sits at a lower index, so the walk
    // strictly decreases and cannot cycle.
    while (nodes_[current.index()].parent.valid()) {
        current = nodes_[current.index()].parent;
    }
    return current;
}

void SceneGraph::mark_subtree(NodeHandle node, std::vector<std::uint32_t>& flags) const {
    flags.assign(nodes_.size(), 0U);
    if (!is_live(node)) {
        return;
    }

    flags[node.index()] = 1U;
    // Starting past the selected node: nothing before it can be beneath it.
    for (std::size_t i = node.index() + 1; i < nodes_.size(); ++i) {
        const NodeHandle parent = nodes_[i].parent;
        if (parent.valid() && flags[parent.index()] != 0U) {
            flags[i] = 1U;
        }
    }
}

SceneNode* SceneGraph::mutable_find(NodeHandle node) {
    return is_live(node) ? &nodes_[node.index()] : nullptr;
}

void SceneGraph::set_mesh(NodeHandle node, const GeometryRegistry::MeshView& mesh,
                          std::uint32_t material_index) {
    SceneNode* target = mutable_find(node);
    SAGE_VERIFY(target != nullptr, "SceneGraph: set_mesh on a stale handle");
    target->mesh = mesh;
    target->material_index = material_index;
    target->has_mesh = true;
}

void SceneGraph::set_local_transform(NodeHandle node, const glm::mat4& local_transform) {
    SceneNode* target = mutable_find(node);
    SAGE_VERIFY(target != nullptr, "SceneGraph: set_local_transform on a stale handle");
    target->local_transform = local_transform;
}

void SceneGraph::update_transforms() {
    for (SceneNode& node : nodes_) {
        // A parent always sits at a lower index, so its world transform is
        // already final by the time its children are reached. That is the entire
        // reason this is a flat loop rather than a recursive traversal.
        node.world_transform =
            node.parent.valid() ? nodes_[node.parent.index()].world_transform * node.local_transform
                                : node.local_transform;
    }
}

void SceneGraph::clear() {
    // Generations are bumped rather than reset, and generations_ keeps its
    // length. Otherwise the next node to land at index 2 would resolve a handle
    // to the old index 2 -- which is precisely the dangling reference the
    // generation counter exists to prevent.
    for (NodeHandle::Generation& generation : generations_) {
        ++generation;
    }
    nodes_.clear();
}

}  // namespace sage::gpu
