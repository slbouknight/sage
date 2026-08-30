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
