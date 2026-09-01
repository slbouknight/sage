#include <sage/gpu/scene.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
