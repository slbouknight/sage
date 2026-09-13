#include <sage/app/viewport_mapping.hpp>
#include <sage/core/math.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using sage::app::ground_plane_hit;
using sage::app::ray_direction;
using sage::app::ViewportMapping;

namespace {

constexpr double k_margin = 1e-5;

// A 1600x900 window whose 3D view occupies the middle, with panels either
// side: the dockspace's central node, which is what viewport_rect_ holds.
ViewportMapping docked_view() {
    ViewportMapping mapping;
    mapping.logical_pos = glm::vec2{0.0F, 0.0F};
    mapping.logical_size = glm::vec2{1600.0F, 900.0F};
    mapping.framebuffer = VkExtent2D{1600, 900};
    mapping.rect = VkRect2D{VkOffset2D{200, 100}, VkExtent2D{1000, 700}};
    return mapping;
}

}  // namespace

// -------------------------------------------------------------- validity ---

TEST_CASE("a default mapping is invalid and converts nothing", "[viewport]") {
    const ViewportMapping mapping;

    CHECK_FALSE(mapping.valid());
    CHECK_FALSE(mapping.to_framebuffer(glm::vec2{10.0F, 10.0F}).has_value());
    CHECK_FALSE(mapping.texel_at(glm::vec2{10.0F, 10.0F}).has_value());
    CHECK_FALSE(mapping.ndc_at(glm::vec2{10.0F, 10.0F}).has_value());
}

TEST_CASE("a zero-size 3D view is invalid", "[viewport]") {
    ViewportMapping mapping = docked_view();
    mapping.rect.extent = VkExtent2D{0, 0};

    // Reachable for real: the window is minimised, or the dockspace has not
    // been laid out yet on the first frame.
    CHECK_FALSE(mapping.valid());
    CHECK_FALSE(mapping.ndc_at(glm::vec2{800.0F, 450.0F}).has_value());
}

// ------------------------------------------------------------ framebuffer ---

TEST_CASE("logical coordinates map 1:1 without display scaling", "[viewport]") {
    const ViewportMapping mapping = docked_view();

    const auto pixel = mapping.to_framebuffer(glm::vec2{700.0F, 450.0F});
    REQUIRE(pixel.has_value());
    CHECK(pixel->x == Approx(700.0F).margin(k_margin));
    CHECK(pixel->y == Approx(450.0F).margin(k_margin));
}

TEST_CASE("logical coordinates scale to a larger framebuffer", "[viewport]") {
    ViewportMapping mapping = docked_view();
    // A 2x HiDPI display: ImGui still reports 1600x900 logical units while the
    // swapchain is 3200x1800.
    mapping.framebuffer = VkExtent2D{3200, 1800};
    mapping.rect = VkRect2D{VkOffset2D{400, 200}, VkExtent2D{2000, 1400}};

    const auto pixel = mapping.to_framebuffer(glm::vec2{700.0F, 450.0F});
    REQUIRE(pixel.has_value());
    CHECK(pixel->x == Approx(1400.0F).margin(k_margin));
    CHECK(pixel->y == Approx(900.0F).margin(k_margin));
}

TEST_CASE("the viewport's own offset is subtracted", "[viewport]") {
    ViewportMapping mapping = docked_view();
    mapping.logical_pos = glm::vec2{50.0F, 20.0F};

    // Always zero today, since multi-viewport is off -- but picking and
    // placement used to disagree about whether it counted, and one of them
    // would have been wrong the moment it stopped being zero.
    const auto pixel = mapping.to_framebuffer(glm::vec2{250.0F, 120.0F});
    REQUIRE(pixel.has_value());
    CHECK(pixel->x == Approx(200.0F).margin(k_margin));
    CHECK(pixel->y == Approx(100.0F).margin(k_margin));
}

// ------------------------------------------------------------------ texel ---

TEST_CASE("a position inside the 3D view resolves to a texel", "[viewport]") {
    const ViewportMapping mapping = docked_view();

    const auto texel = mapping.texel_at(glm::vec2{700.0F, 450.0F});
    REQUIRE(texel.has_value());
    CHECK(texel->x == 700);
    CHECK(texel->y == 450);
}

TEST_CASE("positions outside the 3D view are rejected", "[viewport]") {
    const ViewportMapping mapping = docked_view();

    // Clicking a docked panel or the dockspace border is not a miss in the
    // view, it is not a click in the view at all -- the selection must stand.
    CHECK_FALSE(mapping.texel_at(glm::vec2{100.0F, 450.0F}).has_value());   // left panel
    CHECK_FALSE(mapping.texel_at(glm::vec2{1400.0F, 450.0F}).has_value());  // right panel
    CHECK_FALSE(mapping.texel_at(glm::vec2{700.0F, 50.0F}).has_value());    // menu bar
    CHECK_FALSE(mapping.texel_at(glm::vec2{700.0F, 850.0F}).has_value());   // below
}

TEST_CASE("the 3D view's edges are half-open", "[viewport]") {
    const ViewportMapping mapping = docked_view();
    // rect is offset (200,100), extent 1000x700 -> x in [200, 1200).

    CHECK(mapping.texel_at(glm::vec2{200.0F, 100.0F}).has_value());
    CHECK(mapping.texel_at(glm::vec2{1199.0F, 799.0F}).has_value());
    // One past the last texel is outside; including it would read a pixel the
    // scene pass never wrote.
    CHECK_FALSE(mapping.texel_at(glm::vec2{1200.0F, 450.0F}).has_value());
    CHECK_FALSE(mapping.texel_at(glm::vec2{700.0F, 800.0F}).has_value());
    CHECK_FALSE(mapping.texel_at(glm::vec2{199.0F, 450.0F}).has_value());
}

// -------------------------------------------------------------------- ndc ---

TEST_CASE("the centre of the 3D view is the NDC origin", "[viewport]") {
    const ViewportMapping mapping = docked_view();

    // Centre of rect (200,100)+(1000,700) is (700, 450).
    const auto ndc = mapping.ndc_at(glm::vec2{700.0F, 450.0F});
    REQUIRE(ndc.has_value());
    CHECK(ndc->x == Approx(0.0F).margin(k_margin));
    CHECK(ndc->y == Approx(0.0F).margin(k_margin));
}

TEST_CASE("NDC runs -1 to +1 across the 3D view, with Y downward", "[viewport]") {
    const ViewportMapping mapping = docked_view();

    const auto top_left = mapping.ndc_at(glm::vec2{200.0F, 100.0F});
    REQUIRE(top_left.has_value());
    CHECK(top_left->x == Approx(-1.0F).margin(k_margin));
    // Vulkan's NDC Y runs down the screen exactly as framebuffer rows do, so
    // the top of the view is -1 and there is no negation anywhere.
    CHECK(top_left->y == Approx(-1.0F).margin(k_margin));

    const auto near_bottom_right = mapping.ndc_at(glm::vec2{1199.0F, 799.0F});
    REQUIRE(near_bottom_right.has_value());
    CHECK(near_bottom_right->x == Approx(0.998F).margin(1e-3));
    CHECK(near_bottom_right->y == Approx(0.997F).margin(1e-3));
}

TEST_CASE("NDC agrees with the texel test about what is inside", "[viewport]") {
    const ViewportMapping mapping = docked_view();

    // Picking and placement must not disagree about the view's edge: a click
    // that selects an object has to be a click that can place one.
    for (float x = 150.0F; x < 1300.0F; x += 7.0F) {
        const glm::vec2 point{x, 450.0F};
        INFO("x = " << x);
        CHECK(mapping.texel_at(point).has_value() == mapping.ndc_at(point).has_value());
    }
}

// -------------------------------------------------------------------- ray ---

TEST_CASE("a ray through the centre of the view points along the camera's forward", "[viewport]") {
    // Camera at +Z looking at the origin down -Z.
    const glm::vec3 eye{0.0F, 0.0F, 10.0F};
    const glm::mat4 view = glm::lookAt(eye, glm::vec3(0.0F), glm::vec3(0.0F, 1.0F, 0.0F));
    const glm::mat4 projection =
        sage::core::perspective_vk(glm::radians(60.0F), 16.0F / 9.0F, 0.1F, 100.0F);

    const auto direction =
        ray_direction(glm::inverse(projection * view), eye, glm::vec2{0.0F, 0.0F});
    REQUIRE(direction.has_value());
    CHECK(direction->x == Approx(0.0F).margin(1e-4));
    CHECK(direction->y == Approx(0.0F).margin(1e-4));
    CHECK(direction->z == Approx(-1.0F).margin(1e-4));
    CHECK(glm::length(*direction) == Approx(1.0F));
}

TEST_CASE("a ray off-centre leans the right way", "[viewport]") {
    const glm::vec3 eye{0.0F, 0.0F, 10.0F};
    const glm::mat4 view = glm::lookAt(eye, glm::vec3(0.0F), glm::vec3(0.0F, 1.0F, 0.0F));
    const glm::mat4 projection =
        sage::core::perspective_vk(glm::radians(60.0F), 16.0F / 9.0F, 0.1F, 100.0F);
    const glm::mat4 inverse_vp = glm::inverse(projection * view);

    // +x in NDC is to the right.
    const auto right = ray_direction(inverse_vp, eye, glm::vec2{0.5F, 0.0F});
    REQUIRE(right.has_value());
    CHECK(right->x > 0.0F);

    // +y in NDC is *down* the screen under Vulkan's convention, so the ray
    // must go down in world space too. Getting this backwards would place new
    // objects mirrored about the horizon.
    const auto down = ray_direction(inverse_vp, eye, glm::vec2{0.0F, 0.5F});
    REQUIRE(down.has_value());
    CHECK(down->y < 0.0F);
}

// ----------------------------------------------------------------- ground ---

TEST_CASE("a downward ray meets the ground below its origin", "[viewport]") {
    const auto hit =
        ground_plane_hit(glm::vec3(3.0F, 5.0F, -2.0F), glm::vec3(0.0F, -1.0F, 0.0F), 1000.0F);
    REQUIRE(hit.has_value());
    CHECK(hit->x == Approx(3.0F).margin(k_margin));
    CHECK(hit->y == Approx(0.0F).margin(k_margin));
    CHECK(hit->z == Approx(-2.0F).margin(k_margin));
}

TEST_CASE("a slanted ray meets the ground ahead of its origin", "[viewport]") {
    // From (0, 4, 0) heading down and forward at 45 degrees: 4 units down is
    // 4 units along -Z.
    const glm::vec3 direction = glm::normalize(glm::vec3(0.0F, -1.0F, -1.0F));
    const auto hit = ground_plane_hit(glm::vec3(0.0F, 4.0F, 0.0F), direction, 1000.0F);
    REQUIRE(hit.has_value());
    CHECK(hit->y == Approx(0.0F).margin(k_margin));
    CHECK(hit->z == Approx(-4.0F).margin(1e-4));
}

TEST_CASE("a ray pointing away from the ground misses", "[viewport]") {
    // Looking up at the horizon is the common case, not an error.
    CHECK_FALSE(ground_plane_hit(glm::vec3(0.0F, 5.0F, 0.0F), glm::vec3(0.0F, 1.0F, 0.0F), 1000.0F)
                    .has_value());
}

TEST_CASE("a ray parallel to the ground misses", "[viewport]") {
    CHECK_FALSE(ground_plane_hit(glm::vec3(0.0F, 5.0F, 0.0F), glm::vec3(1.0F, 0.0F, 0.0F), 1000.0F)
                    .has_value());
}

TEST_CASE("a hit beyond the distance limit is rejected", "[viewport]") {
    // A near-horizontal ray meets the plane, but kilometres away -- placing an
    // object out there is never what was meant.
    const glm::vec3 direction = glm::normalize(glm::vec3(0.0F, -0.001F, -1.0F));
    CHECK_FALSE(ground_plane_hit(glm::vec3(0.0F, 5.0F, 0.0F), direction, 1000.0F).has_value());

    // The same ray with a limit that does admit it.
    CHECK(ground_plane_hit(glm::vec3(0.0F, 5.0F, 0.0F), direction, 100000.0F).has_value());
}

TEST_CASE("a ray from below the ground pointing down misses", "[viewport]") {
    // The plane is behind the camera; the intersection is at a negative
    // distance and must not be reported as a hit in front of it.
    CHECK_FALSE(
        ground_plane_hit(glm::vec3(0.0F, -2.0F, 0.0F), glm::vec3(0.0F, -1.0F, 0.0F), 1000.0F)
            .has_value());
}

TEST_CASE("a ray from below the ground pointing up hits it", "[viewport]") {
    const auto hit =
        ground_plane_hit(glm::vec3(1.0F, -2.0F, 0.0F), glm::vec3(0.0F, 1.0F, 0.0F), 1000.0F);
    REQUIRE(hit.has_value());
    CHECK(hit->y == Approx(0.0F).margin(k_margin));
    CHECK(hit->x == Approx(1.0F).margin(k_margin));
}
