#include <sage/core/camera.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;

namespace {

constexpr float k_pi = 3.14159265F;

}  // namespace

TEST_CASE("Zero input leaves the camera unchanged", "[camera]") {
    sage::core::Camera camera({1.0F, 2.0F, 3.0F}, 0.5F, -0.25F);

    for (int frame = 0; frame < 10; ++frame) {
        camera.update({}, 0.016F);
    }

    CHECK(camera.yaw() == Approx(0.5F));
    CHECK(camera.pitch() == Approx(-0.25F));
    CHECK(camera.position().x == Approx(1.0F));
    CHECK(camera.position().y == Approx(2.0F));
    CHECK(camera.position().z == Approx(3.0F));
}

TEST_CASE("Pitch accumulates by the delta, not by itself", "[camera]") {
    sage::core::Camera camera({0.0F, 0.0F, 0.0F}, 0.0F, 0.0F);

    sage::core::CameraInput input;
    input.pitch_delta = 0.1F;
    camera.update(input, 0.016F);

    CHECK(camera.pitch() == Approx(0.1F));
}

TEST_CASE("Pitch clamps short of vertical", "[camera]") {
    sage::core::Camera camera({0.0F, 0.0F, 0.0F}, 0.0F, 0.0F);

    sage::core::CameraInput input;
    input.pitch_delta = 1.0F;
    for (int frame = 0; frame < 10; ++frame) {
        camera.update(input, 0.016F);
    }
    CHECK(camera.pitch() < k_pi / 2.0F);

    input.pitch_delta = -1.0F;
    for (int frame = 0; frame < 20; ++frame) {
        camera.update(input, 0.016F);
    }
    CHECK(camera.pitch() > -k_pi / 2.0F);
}

TEST_CASE("Forward stays a unit vector at extreme pitch", "[camera]") {
    sage::core::Camera camera({0.0F, 0.0F, 0.0F}, 1.0F, 0.0F);

    sage::core::CameraInput input;
    input.pitch_delta = 1.0F;
    for (int frame = 0; frame < 10; ++frame) {
        camera.update(input, 0.016F);
    }

    const glm::vec3 forward = camera.forward();
    CHECK(glm::length(forward) == Approx(1.0F));
    CHECK(std::isfinite(forward.x));
    CHECK(std::isfinite(forward.y));
    CHECK(std::isfinite(forward.z));
}

TEST_CASE("Up and down move along world Y regardless of pitch", "[camera]") {
    sage::core::Camera camera({0.0F, 0.0F, 0.0F}, 0.0F, -1.0F);

    sage::core::CameraInput input;
    input.up = 1.0F;
    camera.update(input, 1.0F);

    // Pitched steeply down, but vertical movement must stay purely vertical.
    CHECK(camera.position().x == Approx(0.0F).margin(1e-5));
    CHECK(camera.position().z == Approx(0.0F).margin(1e-5));
    CHECK(camera.position().y > 0.0F);
}

TEST_CASE("Movement scales with delta time", "[camera]") {
    sage::core::Camera slow({0.0F, 0.0F, 0.0F}, 0.0F, 0.0F);
    sage::core::Camera fast({0.0F, 0.0F, 0.0F}, 0.0F, 0.0F);

    sage::core::CameraInput input;
    input.forward = 1.0F;

    slow.update(input, 0.1F);
    fast.update(input, 0.2F);

    CHECK(glm::length(fast.position()) == Approx(2.0F * glm::length(slow.position())));
}