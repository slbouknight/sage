#include <sage/gpu/scene.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

using Catch::Approx;
using sage::gpu::NodeHandle;
using sage::gpu::SceneGraph;

// add_node's parents-precede-children check is a SAGE_VERIFY, which traps
// rather than throws, so it cannot be exercised from here. What these cases
// cover is the behaviour that invariant exists to produce.

namespace {

const glm::mat4 k_identity{1.0F};

glm::mat4 translation(float x, float y, float z) {
    return glm::translate(k_identity, glm::vec3(x, y, z));
}

// The fourth column of an affine matrix is its translation.
glm::vec3 world_position(const SceneGraph& graph, NodeHandle node) {
    return glm::vec3(graph.find(node)->world_transform[3]);
}

// An absolute margin, not Approx's default relative epsilon: a rotation leaves
// residue on the order of 1e-7 in components that should be zero, and nothing
// is *relatively* close to zero.
constexpr double k_margin = 1e-5;

void check_position(const glm::vec3& actual, float x, float y, float z) {
    CHECK(actual.x == Approx(x).margin(k_margin));
    CHECK(actual.y == Approx(y).margin(k_margin));
    CHECK(actual.z == Approx(z).margin(k_margin));
}

}  // namespace

TEST_CASE("A root's world transform is its local transform", "[scene]") {
    SceneGraph graph;
    const NodeHandle root = graph.add_node(NodeHandle{}, translation(10.0F, 0.0F, 0.0F), "root");

    graph.update_transforms();

    check_position(world_position(graph, root), 10.0F, 0.0F, 0.0F);
}

TEST_CASE("World transforms compose down a chain", "[scene]") {
    SceneGraph graph;
    const NodeHandle root = graph.add_node(NodeHandle{}, translation(10.0F, 0.0F, 0.0F), "root");
    const NodeHandle child = graph.add_node(root, translation(0.0F, 5.0F, 0.0F), "child");
    const NodeHandle grandchild =
        graph.add_node(child, translation(0.0F, 0.0F, 2.0F), "grandchild");

    graph.update_transforms();

    check_position(world_position(graph, grandchild), 10.0F, 5.0F, 2.0F);
}

TEST_CASE("A parent's rotation carries its children around it", "[scene]") {
    SceneGraph graph;
    // Half a turn about Y maps (x, y, z) to (-x, y, -z) -- the same transform
    // Lantern's root carries, and the one that made object-space normals wrong.
    const glm::mat4 half_turn = glm::rotate(k_identity, glm::radians(180.0F), glm::vec3(0, 1, 0));
    const NodeHandle root = graph.add_node(NodeHandle{}, half_turn, "root");
    const NodeHandle child = graph.add_node(root, translation(3.0F, 0.0F, 0.0F), "child");

    graph.update_transforms();

    check_position(world_position(graph, child), -3.0F, 0.0F, 0.0F);
}

TEST_CASE("Siblings compose independently", "[scene]") {
    SceneGraph graph;
    const NodeHandle root = graph.add_node(NodeHandle{}, translation(1.0F, 0.0F, 0.0F), "root");
    const NodeHandle left = graph.add_node(root, translation(0.0F, 2.0F, 0.0F), "left");
    const NodeHandle right = graph.add_node(root, translation(0.0F, 0.0F, 3.0F), "right");

    graph.update_transforms();

    check_position(world_position(graph, left), 1.0F, 2.0F, 0.0F);
    check_position(world_position(graph, right), 1.0F, 0.0F, 3.0F);
}

TEST_CASE("Moving a parent moves its descendants", "[scene]") {
    SceneGraph graph;
    const NodeHandle root = graph.add_node(NodeHandle{}, k_identity, "root");
    const NodeHandle child = graph.add_node(root, translation(0.0F, 1.0F, 0.0F), "child");

    graph.set_local_transform(root, translation(100.0F, 0.0F, 0.0F));
    graph.update_transforms();

    check_position(world_position(graph, child), 100.0F, 1.0F, 0.0F);
}

TEST_CASE("A default-constructed handle never resolves", "[scene]") {
    SceneGraph graph;
    CHECK(graph.find(NodeHandle{}) == nullptr);

    graph.add_node(NodeHandle{}, k_identity, "root");
    CHECK(graph.find(NodeHandle{}) == nullptr);
}

TEST_CASE("Nodes carry a mesh only once one is set", "[scene]") {
    SceneGraph graph;
    const NodeHandle node = graph.add_node(NodeHandle{}, k_identity, "node");
    CHECK_FALSE(graph.find(node)->has_mesh);

    sage::gpu::GeometryRegistry::MeshView mesh;
    mesh.index_count = 36;
    graph.set_mesh(node, mesh, 7);

    CHECK(graph.find(node)->has_mesh);
    CHECK(graph.find(node)->mesh.index_count == 36);
    CHECK(graph.find(node)->material_index == 7);
}

TEST_CASE("clear drops every node", "[scene]") {
    SceneGraph graph;
    graph.add_node(NodeHandle{}, k_identity, "a");
    graph.add_node(NodeHandle{}, k_identity, "b");
    CHECK(graph.size() == 2);

    graph.clear();

    CHECK(graph.empty());
    CHECK(graph.size() == 0);
}

TEST_CASE("An index reused after clear does not resolve the old handle", "[scene]") {
    SceneGraph graph;
    const NodeHandle before = graph.add_node(NodeHandle{}, k_identity, "before");

    graph.clear();
    const NodeHandle after = graph.add_node(NodeHandle{}, k_identity, "after");

    // The slot is genuinely recycled -- which is exactly why the generation
    // counter has to move, and this is the case that proves it does.
    CHECK(after.index() == before.index());
    CHECK(after.generation() != before.generation());
    CHECK(graph.find(before) == nullptr);
    REQUIRE(graph.find(after) != nullptr);
    CHECK(graph.find(after)->name == "after");
}

TEST_CASE("Handles stay valid as the node array grows", "[scene]") {
    SceneGraph graph;
    const NodeHandle first = graph.add_node(NodeHandle{}, translation(1.0F, 0.0F, 0.0F), "first");

    // Enough to force the underlying vector to reallocate more than once.
    for (int i = 0; i < 256; ++i) {
        graph.add_node(NodeHandle{}, k_identity, "filler");
    }

    graph.update_transforms();

    REQUIRE(graph.find(first) != nullptr);
    check_position(world_position(graph, first), 1.0F, 0.0F, 0.0F);
}

TEST_CASE("handle_at is the inverse of the node array index", "[scene]") {
    SceneGraph graph;
    const NodeHandle root = graph.add_node(NodeHandle{}, k_identity, "root");
    const NodeHandle child = graph.add_node(root, k_identity, "child");

    // What object-id picking depends on: the GPU hands back an index, and this
    // has to name the same node the draw loop wrote that index for.
    CHECK(graph.handle_at(0) == root);
    CHECK(graph.handle_at(1) == child);

    REQUIRE(graph.find(graph.handle_at(1)) != nullptr);
    CHECK(graph.find(graph.handle_at(1))->name == "child");
}

TEST_CASE("handle_at rejects an out-of-range index", "[scene]") {
    SceneGraph graph;
    graph.add_node(NodeHandle{}, k_identity, "only");

    // A pick that lands on a stale id -- the scene shrank between the draw and
    // the readback -- must come back invalid rather than naming a live node.
    CHECK_FALSE(graph.handle_at(1).valid());
    CHECK_FALSE(graph.handle_at(99).valid());
    CHECK(graph.find(graph.handle_at(1)) == nullptr);
}

TEST_CASE("handle_at after a clear does not resolve a pre-clear handle", "[scene]") {
    SceneGraph graph;
    const NodeHandle before = graph.add_node(NodeHandle{}, k_identity, "before");

    graph.clear();
    graph.add_node(NodeHandle{}, k_identity, "after");

    // Index 0 is occupied again, but by a different node. A pick resolved
    // through handle_at must not compare equal to the handle issued earlier,
    // which is what stops a selection surviving a scene reload.
    const NodeHandle reused = graph.handle_at(0);
    REQUIRE(reused.valid());
    CHECK(reused != before);
    REQUIRE(graph.find(reused) != nullptr);
    CHECK(graph.find(reused)->name == "after");
}

TEST_CASE("root_of walks to the topmost ancestor", "[scene]") {
    SceneGraph graph;
    const NodeHandle root = graph.add_node(NodeHandle{}, k_identity, "root");
    const NodeHandle middle = graph.add_node(root, k_identity, "middle");
    const NodeHandle leaf = graph.add_node(middle, k_identity, "leaf");

    // What a viewport click relies on: any depth resolves to the object.
    CHECK(graph.root_of(leaf) == root);
    CHECK(graph.root_of(middle) == root);
    CHECK(graph.root_of(root) == root);
    CHECK_FALSE(graph.root_of(NodeHandle{}).valid());
}

TEST_CASE("mark_subtree flags a node and everything beneath it", "[scene]") {
    SceneGraph graph;
    const NodeHandle root = graph.add_node(NodeHandle{}, k_identity, "root");
    const NodeHandle middle = graph.add_node(root, k_identity, "middle");
    const NodeHandle leaf = graph.add_node(middle, k_identity, "leaf");
    const NodeHandle other = graph.add_node(NodeHandle{}, k_identity, "other");

    std::vector<std::uint32_t> flags;

    graph.mark_subtree(root, flags);
    REQUIRE(flags.size() == 4);
    CHECK(flags[root.index()] == 1);
    CHECK(flags[middle.index()] == 1);
    CHECK(flags[leaf.index()] == 1);
    CHECK(flags[other.index()] == 0);

    // Selecting a middle node must catch its descendants but not its parent,
    // which is the case a contiguous index range would get wrong.
    graph.mark_subtree(middle, flags);
    CHECK(flags[root.index()] == 0);
    CHECK(flags[middle.index()] == 1);
    CHECK(flags[leaf.index()] == 1);
    CHECK(flags[other.index()] == 0);

    graph.mark_subtree(NodeHandle{}, flags);
    CHECK(std::ranges::none_of(flags, [](std::uint32_t f) { return f != 0; }));
}

TEST_CASE("mark_subtree does not rely on descendants being contiguous", "[scene]") {
    SceneGraph graph;
    const NodeHandle a = graph.add_node(NodeHandle{}, k_identity, "a");
    const NodeHandle b = graph.add_node(NodeHandle{}, k_identity, "b");
    // Interleaved: a's child is added after b, so a's subtree is indices 0 and
    // 2 with an unrelated node sitting between them.
    const NodeHandle a_child = graph.add_node(a, k_identity, "a_child");

    std::vector<std::uint32_t> flags;
    graph.mark_subtree(a, flags);

    CHECK(flags[a.index()] == 1);
    CHECK(flags[b.index()] == 0);
    CHECK(flags[a_child.index()] == 1);
}

TEST_CASE("remove_subtree takes a node and its descendants", "[scene_graph]") {
    sage::gpu::SceneGraph graph;
    const auto root = graph.add_node({}, k_identity, "root");
    const auto child = graph.add_node(root, k_identity, "child");
    const auto grandchild = graph.add_node(child, k_identity, "grandchild");
    const auto sibling = graph.add_node({}, k_identity, "sibling");

    graph.remove_subtree(child);

    CHECK(graph.find(child) == nullptr);
    CHECK(graph.find(grandchild) == nullptr);
    // Neither the ancestor nor an unrelated root goes with it.
    CHECK(graph.find(root) != nullptr);
    CHECK(graph.find(sibling) != nullptr);
}

TEST_CASE("removal leaves surviving handles resolving to the same nodes", "[scene_graph]") {
    sage::gpu::SceneGraph graph;
    const auto first = graph.add_node({}, k_identity, "first");
    const auto second = graph.add_node({}, k_identity, "second");
    const auto third = graph.add_node({}, k_identity, "third");

    graph.remove_subtree(first);

    // The whole reason deletion tombstones rather than compacting. Erasing
    // index 0 would slide these down and leave both handles naming the wrong
    // node -- silently, since the generations would still match.
    REQUIRE(graph.find(second) != nullptr);
    REQUIRE(graph.find(third) != nullptr);
    CHECK(graph.find(second)->name == "second");
    CHECK(graph.find(third)->name == "third");
}

TEST_CASE("a deleted node is distinguishable from a stale handle", "[scene_graph]") {
    sage::gpu::SceneGraph graph;
    const auto node = graph.add_node({}, k_identity, "node");

    CHECK_FALSE(graph.is_deleted(node));
    graph.remove_subtree(node);
    // Deleted: gone from find, but still resolvable, which is what lets undo
    // hand the same handle to restore_subtree.
    CHECK(graph.is_deleted(node));
    CHECK(graph.find(node) == nullptr);

    graph.clear();
    // Cleared: the generation moved, so the handle names nothing at all.
    CHECK_FALSE(graph.is_deleted(node));
}

TEST_CASE("restore_subtree puts back exactly what was removed", "[scene_graph]") {
    sage::gpu::SceneGraph graph;
    const auto root = graph.add_node({}, k_identity, "root");
    const auto child = graph.add_node(root, k_identity, "child");
    const auto grandchild = graph.add_node(child, k_identity, "grandchild");

    graph.remove_subtree(child);
    graph.restore_subtree(child);

    CHECK(graph.find(child) != nullptr);
    CHECK(graph.find(grandchild) != nullptr);
    CHECK(graph.live_size() == 3);
}

TEST_CASE("live_size counts nodes, size counts slots", "[scene_graph]") {
    sage::gpu::SceneGraph graph;
    const auto a = graph.add_node({}, k_identity, "a");
    graph.add_node({}, k_identity, "b");

    CHECK(graph.size() == 2);
    CHECK(graph.live_size() == 2);

    graph.remove_subtree(a);
    // The slot stays, which is what keeps b's index -- and so its object id --
    // from moving.
    CHECK(graph.size() == 2);
    CHECK(graph.live_size() == 1);
}

TEST_CASE("mark_subtree ignores deleted nodes", "[scene_graph]") {
    sage::gpu::SceneGraph graph;
    const auto root = graph.add_node({}, k_identity, "root");
    const auto child = graph.add_node(root, k_identity, "child");

    std::vector<std::uint32_t> flags;
    graph.remove_subtree(child);
    graph.mark_subtree(root, flags);

    REQUIRE(flags.size() == 2);
    CHECK(flags[0] == 1U);
    // Otherwise the outline would trace the silhouette of something deleted.
    CHECK(flags[1] == 0U);
}

TEST_CASE("root_of refuses a deleted node", "[scene_graph]") {
    sage::gpu::SceneGraph graph;
    const auto root = graph.add_node({}, k_identity, "root");
    const auto child = graph.add_node(root, k_identity, "child");

    graph.remove_subtree(root);
    CHECK_FALSE(graph.root_of(child).valid());
}
