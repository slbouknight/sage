#include <sage/app/scene_query.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <numbers>
#include <vector>

using Catch::Approx;
using sage::app::Bounds;
using sage::app::build_child_table;
using sage::app::collect_lights;
using sage::app::compute_scene_bounds;
using sage::app::fit_directional_light;
using sage::gpu::Light;
using sage::gpu::LightType;
using sage::gpu::NodeHandle;
using sage::gpu::SceneGraph;
using sage::gpu::SceneLight;

namespace {

// A rotation leaves residue on the order of 1e-7 in components that should be
// zero, and nothing is *relatively* close to zero -- so an absolute margin.
constexpr double k_margin = 1e-5;

void check_vec(const glm::vec3& actual, float x, float y, float z) {
    CHECK(actual.x == Approx(x).margin(k_margin));
    CHECK(actual.y == Approx(y).margin(k_margin));
    CHECK(actual.z == Approx(z).margin(k_margin));
}

// A unit cube centred on the origin, in the node's own space.
sage::gpu::GeometryRegistry::MeshView unit_cube() {
    sage::gpu::GeometryRegistry::MeshView mesh;
    // Non-zero so the view counts as drawable; the query only reads bounds.
    mesh.index_count = 36;
    mesh.bounds_min = glm::vec3(-0.5F);
    mesh.bounds_max = glm::vec3(0.5F);
    return mesh;
}

NodeHandle add_cube(SceneGraph& graph, const glm::mat4& transform, std::string name = "cube") {
    const NodeHandle node = graph.add_node(NodeHandle{}, transform, std::move(name));
    graph.set_mesh(node, unit_cube(), 0);
    return node;
}

NodeHandle add_light(SceneGraph& graph, const glm::mat4& transform, const SceneLight& light) {
    const NodeHandle node = graph.add_node(NodeHandle{}, transform, "light");
    graph.set_light(node, light);
    return node;
}

}  // namespace

// ---------------------------------------------------------------- bounds ---

TEST_CASE("an empty graph has empty bounds", "[scene_query]") {
    const SceneGraph graph;
    CHECK(compute_scene_bounds(graph).empty());
}

TEST_CASE("a graph of transform-only nodes has empty bounds", "[scene_query]") {
    SceneGraph graph;
    graph.add_node(NodeHandle{}, glm::translate(glm::mat4{1.0F}, glm::vec3(10.0F)), "empty");
    graph.update_transforms();

    // A node with no mesh contributes nothing, however far out it sits.
    CHECK(compute_scene_bounds(graph).empty());
}

TEST_CASE("bounds cover a single mesh at the origin", "[scene_query]") {
    SceneGraph graph;
    add_cube(graph, glm::mat4{1.0F});
    graph.update_transforms();

    const Bounds bounds = compute_scene_bounds(graph);
    REQUIRE_FALSE(bounds.empty());
    check_vec(bounds.min, -0.5F, -0.5F, -0.5F);
    check_vec(bounds.max, 0.5F, 0.5F, 0.5F);
}

TEST_CASE("bounds follow a node's world transform", "[scene_query]") {
    SceneGraph graph;
    add_cube(graph, glm::translate(glm::mat4{1.0F}, glm::vec3(10.0F, 0.0F, 0.0F)));
    graph.update_transforms();

    const Bounds bounds = compute_scene_bounds(graph);
    check_vec(bounds.min, 9.5F, -0.5F, -0.5F);
    check_vec(bounds.max, 10.5F, 0.5F, 0.5F);
}

TEST_CASE("bounds enclose a rotated mesh", "[scene_query]") {
    SceneGraph graph;
    // 45 degrees about Y. Transforming only min and max would give a box of
    // the original size; the true extent in X and Z is half the cube's
    // diagonal, sqrt(2)/2 ~ 0.7071.
    add_cube(graph, glm::rotate(glm::mat4{1.0F}, std::numbers::pi_v<float> / 4.0F,
                                glm::vec3(0.0F, 1.0F, 0.0F)));
    graph.update_transforms();

    const Bounds bounds = compute_scene_bounds(graph);
    const float diagonal = std::numbers::sqrt2_v<float> / 2.0F;
    check_vec(bounds.min, -diagonal, -0.5F, -diagonal);
    check_vec(bounds.max, diagonal, 0.5F, diagonal);
}

TEST_CASE("bounds span every mesh in the graph", "[scene_query]") {
    SceneGraph graph;
    add_cube(graph, glm::translate(glm::mat4{1.0F}, glm::vec3(-4.0F, 0.0F, 0.0F)), "left");
    add_cube(graph, glm::translate(glm::mat4{1.0F}, glm::vec3(4.0F, 2.0F, 0.0F)), "right");
    graph.update_transforms();

    const Bounds bounds = compute_scene_bounds(graph);
    check_vec(bounds.min, -4.5F, -0.5F, -0.5F);
    check_vec(bounds.max, 4.5F, 2.5F, 0.5F);
}

TEST_CASE("bounds inherit a parent's transform", "[scene_query]") {
    SceneGraph graph;
    const NodeHandle parent = graph.add_node(
        NodeHandle{}, glm::translate(glm::mat4{1.0F}, glm::vec3(0.0F, 5.0F, 0.0F)), "parent");
    const NodeHandle child = graph.add_node(parent, glm::mat4{1.0F}, "child");
    graph.set_mesh(child, unit_cube(), 0);
    graph.update_transforms();

    const Bounds bounds = compute_scene_bounds(graph);
    check_vec(bounds.min, -0.5F, 4.5F, -0.5F);
    check_vec(bounds.max, 0.5F, 5.5F, 0.5F);
}

TEST_CASE("editor-only meshes are excluded from bounds", "[scene_query]") {
    SceneGraph graph;
    add_cube(graph, glm::mat4{1.0F}, "subject");
    // A light icon floating well above the subject. Counting it would stretch
    // the shadow frustum over empty air and resize every primitive added next.
    const NodeHandle icon =
        add_cube(graph, glm::translate(glm::mat4{1.0F}, glm::vec3(0.0F, 100.0F, 0.0F)), "icon");
    graph.set_editor_only(icon, true);
    graph.update_transforms();

    const Bounds bounds = compute_scene_bounds(graph);
    check_vec(bounds.max, 0.5F, 0.5F, 0.5F);
}

TEST_CASE("deleted meshes are excluded from bounds", "[scene_query]") {
    SceneGraph graph;
    add_cube(graph, glm::mat4{1.0F}, "kept");
    const NodeHandle gone =
        add_cube(graph, glm::translate(glm::mat4{1.0F}, glm::vec3(50.0F, 0.0F, 0.0F)), "gone");
    graph.update_transforms();

    graph.remove_subtree(gone);
    const Bounds bounds = compute_scene_bounds(graph);
    check_vec(bounds.max, 0.5F, 0.5F, 0.5F);

    // And restoring it brings the bounds back, since nothing was discarded.
    graph.restore_subtree(gone);
    check_vec(compute_scene_bounds(graph).max, 50.5F, 0.5F, 0.5F);
}

TEST_CASE("a scene of only editor meshes reads as empty", "[scene_query]") {
    SceneGraph graph;
    const NodeHandle icon = add_cube(graph, glm::mat4{1.0F}, "icon");
    graph.set_editor_only(icon, true);
    graph.update_transforms();

    // Not a zero-size box at the origin: "nothing to frame" has to be
    // distinguishable, because fit_directional_light branches on it.
    CHECK(compute_scene_bounds(graph).empty());
}

// ------------------------------------------------------------- light fit ---

TEST_CASE("fitting empty bounds yields a default fit", "[scene_query]") {
    const Bounds empty{glm::vec3(1.0F), glm::vec3(-1.0F)};
    REQUIRE(empty.empty());

    const auto fit = fit_directional_light(empty, glm::vec3(0.0F, -1.0F, 0.0F), 2048);
    CHECK(fit.world_texel_size == 0.0F);
    CHECK(fit.view_projection == glm::mat4{1.0F});
}

TEST_CASE("the fitted frustum contains the bounds it was fitted to", "[scene_query]") {
    const Bounds bounds{glm::vec3(-2.0F, 0.0F, -2.0F), glm::vec3(2.0F, 3.0F, 2.0F)};
    const auto fit = fit_directional_light(bounds, glm::vec3(-0.8F, -0.5F, -0.33F), 2048);

    // Every corner must land inside clip space. This is the property the
    // shadow map depends on: a corner outside it is geometry that silently
    // stops casting.
    for (int corner = 0; corner < 8; ++corner) {
        const glm::vec3 point{(corner & 1) != 0 ? bounds.max.x : bounds.min.x,
                              (corner & 2) != 0 ? bounds.max.y : bounds.min.y,
                              (corner & 4) != 0 ? bounds.max.z : bounds.min.z};
        const glm::vec4 clip = fit.view_projection * glm::vec4(point, 1.0F);
        INFO("corner " << corner);
        CHECK(clip.x >= Approx(-clip.w).margin(k_margin));
        CHECK(clip.x <= Approx(clip.w).margin(k_margin));
        CHECK(clip.y >= Approx(-clip.w).margin(k_margin));
        CHECK(clip.y <= Approx(clip.w).margin(k_margin));
        // Vulkan's depth range is [0, w], not [-w, w].
        CHECK(clip.z >= Approx(0.0F).margin(k_margin));
        CHECK(clip.z <= Approx(clip.w).margin(k_margin));
    }
}

TEST_CASE("a light pointing straight down still produces a usable frustum", "[scene_query]") {
    const Bounds bounds{glm::vec3(-1.0F), glm::vec3(1.0F)};
    // World up is parallel to the light here, so lookAt's up hint has to be
    // swapped or the view matrix degenerates into NaNs.
    const auto fit = fit_directional_light(bounds, glm::vec3(0.0F, -1.0F, 0.0F), 1024);

    const glm::vec4 clip = fit.view_projection * glm::vec4(0.0F, 0.0F, 0.0F, 1.0F);
    CHECK(std::isfinite(clip.x));
    CHECK(std::isfinite(clip.y));
    CHECK(std::isfinite(clip.z));
    CHECK(std::isfinite(clip.w));
    CHECK(fit.world_texel_size > 0.0F);
}

TEST_CASE("texel size scales with the scene and with resolution", "[scene_query]") {
    const Bounds small{glm::vec3(-1.0F), glm::vec3(1.0F)};
    const Bounds large{glm::vec3(-10.0F), glm::vec3(10.0F)};
    const glm::vec3 direction{0.0F, -1.0F, -1.0F};

    const float small_texel = fit_directional_light(small, direction, 1024).world_texel_size;
    const float large_texel = fit_directional_light(large, direction, 1024).world_texel_size;
    // Ten times the scene across the same number of texels.
    CHECK(large_texel == Approx(small_texel * 10.0F).epsilon(1e-4));

    // Twice the texels across the same scene is half the world per texel --
    // which is why the normal-offset bias is expressed in texels rather than
    // world units.
    const float dense = fit_directional_light(small, direction, 2048).world_texel_size;
    CHECK(dense == Approx(small_texel / 2.0F).epsilon(1e-4));
}

TEST_CASE("a degenerate single-point scene does not divide by zero", "[scene_query]") {
    const Bounds point{glm::vec3(3.0F), glm::vec3(3.0F)};
    REQUIRE_FALSE(point.empty());

    const auto fit = fit_directional_light(point, glm::vec3(0.0F, -1.0F, 0.0F), 2048);
    CHECK(fit.world_texel_size > 0.0F);
    CHECK(std::isfinite(fit.world_texel_size));
}

TEST_CASE("the fit is stable as the light swings around", "[scene_query]") {
    const Bounds bounds{glm::vec3(-2.0F, -1.0F, -3.0F), glm::vec3(2.0F, 1.0F, 3.0F)};

    // Fitted to the bounding sphere rather than the box, so the frustum's size
    // -- and therefore the shadow's resolution -- must not change with the
    // light's direction. Otherwise dragging the direction slider would visibly
    // change shadow quality.
    const float first =
        fit_directional_light(bounds, glm::vec3(-1.0F, -1.0F, 0.0F), 2048).world_texel_size;
    const float second =
        fit_directional_light(bounds, glm::vec3(0.3F, -1.0F, 0.8F), 2048).world_texel_size;
    CHECK(second == Approx(first));
}

// ----------------------------------------------------------------- lights ---

TEST_CASE("a graph with no lights collects none", "[scene_query]") {
    SceneGraph graph;
    add_cube(graph, glm::mat4{1.0F});
    graph.update_transforms();

    std::array<Light, 4> lights{};
    CHECK(collect_lights(graph, lights) == 0);
}

TEST_CASE("a light's position comes from its world transform", "[scene_query]") {
    SceneGraph graph;
    SceneLight authored;
    authored.type = LightType::point;
    authored.intensity = 42.0F;
    authored.range = 7.0F;
    authored.color = glm::vec3(0.25F, 0.5F, 0.75F);
    add_light(graph, glm::translate(glm::mat4{1.0F}, glm::vec3(1.0F, 2.0F, 3.0F)), authored);
    graph.update_transforms();

    std::array<Light, 4> lights{};
    REQUIRE(collect_lights(graph, lights) == 1);

    check_vec(lights[0].position, 1.0F, 2.0F, 3.0F);
    CHECK(lights[0].intensity == 42.0F);
    CHECK(lights[0].range == 7.0F);
    CHECK(lights[0].type == LightType::point);
    check_vec(lights[0].color, 0.25F, 0.5F, 0.75F);
}

TEST_CASE("an unrotated light points straight down", "[scene_query]") {
    SceneGraph graph;
    add_light(graph, glm::mat4{1.0F}, SceneLight{});
    graph.update_transforms();

    std::array<Light, 4> lights{};
    REQUIRE(collect_lights(graph, lights) == 1);
    // -Y is the canonical direction, so the rotate gizmo tilts from there.
    check_vec(lights[0].direction, 0.0F, -1.0F, 0.0F);
}

TEST_CASE("a rotated light's direction follows the node", "[scene_query]") {
    SceneGraph graph;
    // 90 degrees about Z takes -Y to +X.
    add_light(
        graph,
        glm::rotate(glm::mat4{1.0F}, std::numbers::pi_v<float> / 2.0F, glm::vec3(0.0F, 0.0F, 1.0F)),
        SceneLight{});
    graph.update_transforms();

    std::array<Light, 4> lights{};
    REQUIRE(collect_lights(graph, lights) == 1);
    check_vec(lights[0].direction, 1.0F, 0.0F, 0.0F);
}

TEST_CASE("a scaled light still reports a unit direction", "[scene_query]") {
    SceneGraph graph;
    // A non-unit direction would be invisible to the BRDF and would scale
    // every dot product it appears in.
    add_light(graph, glm::scale(glm::mat4{1.0F}, glm::vec3(5.0F)), SceneLight{});
    graph.update_transforms();

    std::array<Light, 4> lights{};
    REQUIRE(collect_lights(graph, lights) == 1);
    CHECK(glm::length(lights[0].direction) == Approx(1.0F));
    check_vec(lights[0].direction, 0.0F, -1.0F, 0.0F);
}

TEST_CASE("a zero-scaled light falls back to pointing down", "[scene_query]") {
    SceneGraph graph;
    add_light(graph, glm::scale(glm::mat4{1.0F}, glm::vec3(0.0F)), SceneLight{});
    graph.update_transforms();

    std::array<Light, 4> lights{};
    REQUIRE(collect_lights(graph, lights) == 1);
    // Normalising a zero vector is a division by zero; the guard is what keeps
    // NaNs out of the frame buffer.
    CHECK(glm::length(lights[0].direction) == Approx(1.0F));
    check_vec(lights[0].direction, 0.0F, -1.0F, 0.0F);
}

TEST_CASE("deleted lights are not collected", "[scene_query]") {
    SceneGraph graph;
    add_light(graph, glm::mat4{1.0F}, SceneLight{});
    const NodeHandle gone = add_light(graph, glm::mat4{1.0F}, SceneLight{});
    graph.update_transforms();

    graph.remove_subtree(gone);
    std::array<Light, 4> lights{};
    CHECK(collect_lights(graph, lights) == 1);
}

TEST_CASE("collection stops at the output's capacity", "[scene_query]") {
    SceneGraph graph;
    for (int i = 0; i < 6; ++i) {
        add_light(graph,
                  glm::translate(glm::mat4{1.0F}, glm::vec3(static_cast<float>(i), 0.0F, 0.0F)),
                  SceneLight{});
    }
    graph.update_transforms();

    // The shader declares a fixed array, so the excess is dropped rather than
    // written past the end.
    std::array<Light, 3> lights{};
    CHECK(collect_lights(graph, lights) == 3);
    // The first three in graph order, not an arbitrary three.
    check_vec(lights[0].position, 0.0F, 0.0F, 0.0F);
    check_vec(lights[2].position, 2.0F, 0.0F, 0.0F);
}

TEST_CASE("an empty output span collects nothing without writing", "[scene_query]") {
    SceneGraph graph;
    add_light(graph, glm::mat4{1.0F}, SceneLight{});
    graph.update_transforms();

    CHECK(collect_lights(graph, std::span<Light>{}) == 0);
}

// ------------------------------------------------------------ child table ---

TEST_CASE("an empty graph has an empty child table", "[scene_query]") {
    const SceneGraph graph;
    const auto table = build_child_table(graph);

    CHECK(table.children.empty());
    CHECK(table.roots.empty());
}

TEST_CASE("unparented nodes are all roots", "[scene_query]") {
    SceneGraph graph;
    add_cube(graph, glm::mat4{1.0F}, "a");
    add_cube(graph, glm::mat4{1.0F}, "b");

    const auto table = build_child_table(graph);
    CHECK(table.roots == std::vector<std::uint32_t>{0, 1});
    CHECK(table.children[0].empty());
    CHECK(table.children[1].empty());
}

TEST_CASE("children are listed under their parent, not as roots", "[scene_query]") {
    SceneGraph graph;
    const NodeHandle parent = graph.add_node(NodeHandle{}, glm::mat4{1.0F}, "parent");
    graph.add_node(parent, glm::mat4{1.0F}, "first");
    graph.add_node(parent, glm::mat4{1.0F}, "second");

    const auto table = build_child_table(graph);
    CHECK(table.roots == std::vector<std::uint32_t>{0});
    // Index order, which is also the order they were added -- the hierarchy
    // panel should not reshuffle rows between frames.
    CHECK(table.children[0] == std::vector<std::uint32_t>{1, 2});
}

TEST_CASE("the table spans several levels", "[scene_query]") {
    SceneGraph graph;
    const NodeHandle root = graph.add_node(NodeHandle{}, glm::mat4{1.0F}, "root");
    const NodeHandle mid = graph.add_node(root, glm::mat4{1.0F}, "mid");
    graph.add_node(mid, glm::mat4{1.0F}, "leaf");

    const auto table = build_child_table(graph);
    CHECK(table.roots == std::vector<std::uint32_t>{0});
    CHECK(table.children[0] == std::vector<std::uint32_t>{1});
    CHECK(table.children[1] == std::vector<std::uint32_t>{2});
    CHECK(table.children[2].empty());
}

TEST_CASE("the table is sized to every slot, tombstones included", "[scene_query]") {
    SceneGraph graph;
    add_cube(graph, glm::mat4{1.0F}, "a");
    const NodeHandle gone = add_cube(graph, glm::mat4{1.0F}, "b");
    graph.remove_subtree(gone);

    // Indexed by node index, so it must cover slots rather than live nodes --
    // otherwise children[i] would be out of range for any i past a tombstone.
    const auto table = build_child_table(graph);
    CHECK(table.children.size() == graph.size());
    CHECK(table.children.size() == 2);
}

TEST_CASE("a deleted node is neither a root nor a child", "[scene_query]") {
    SceneGraph graph;
    add_cube(graph, glm::mat4{1.0F}, "kept");
    const NodeHandle gone = add_cube(graph, glm::mat4{1.0F}, "gone");

    graph.remove_subtree(gone);
    const auto table = build_child_table(graph);
    CHECK(table.roots == std::vector<std::uint32_t>{0});

    // And undo puts the row back.
    graph.restore_subtree(gone);
    CHECK(build_child_table(graph).roots == std::vector<std::uint32_t>{0, 1});
}

TEST_CASE("deleting a parent takes its children out of the table too", "[scene_query]") {
    SceneGraph graph;
    const NodeHandle parent = graph.add_node(NodeHandle{}, glm::mat4{1.0F}, "parent");
    graph.add_node(parent, glm::mat4{1.0F}, "child");

    // remove_subtree tombstones the whole subtree, which is what stops a live
    // child being filed under a dead parent -- where the tree walk, starting
    // from roots, would never reach it and the row would silently vanish.
    graph.remove_subtree(parent);

    const auto table = build_child_table(graph);
    CHECK(table.roots.empty());
    CHECK(table.children[0].empty());
}

TEST_CASE("every live node is reachable from some root", "[scene_query]") {
    SceneGraph graph;
    const NodeHandle a = graph.add_node(NodeHandle{}, glm::mat4{1.0F}, "a");
    const NodeHandle b = graph.add_node(a, glm::mat4{1.0F}, "b");
    graph.add_node(b, glm::mat4{1.0F}, "c");
    graph.add_node(NodeHandle{}, glm::mat4{1.0F}, "d");
    const NodeHandle gone = graph.add_node(NodeHandle{}, glm::mat4{1.0F}, "gone");
    graph.remove_subtree(gone);

    // The property the hierarchy panel depends on: walking from roots visits
    // every live node exactly once. A node the walk cannot reach is a row the
    // user cannot see or select.
    std::vector<std::uint32_t> visited;
    const auto table = build_child_table(graph);
    const auto walk = [&](auto&& self, std::uint32_t index) -> void {
        visited.push_back(index);
        for (const std::uint32_t child : table.children[index]) {
            self(self, child);
        }
    };
    for (const std::uint32_t root : table.roots) {
        walk(walk, root);
    }

    std::sort(visited.begin(), visited.end());
    CHECK(visited == std::vector<std::uint32_t>{0, 1, 2, 3});
    CHECK(visited.size() == graph.live_size());
}
