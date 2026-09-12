#include <sage/gpu/drag_gesture.hpp>

#include <catch2/catch_test_macros.hpp>

using sage::gpu::DragGesture;

namespace {

// The cursor does not move between frames in most of these, so a helper that
// holds the position steady keeps the interesting part of each test visible.
DragGesture::Update hold(DragGesture& gesture, bool down, double x = 100.0, double y = 100.0) {
    return gesture.update(down, x, y);
}

}  // namespace

TEST_CASE("a press and release without movement is a click", "[drag_gesture]") {
    DragGesture gesture;

    const DragGesture::Update pressed = hold(gesture, true);
    CHECK_FALSE(pressed.look_active);
    CHECK_FALSE(pressed.click);
    CHECK_FALSE(pressed.began_look);

    const DragGesture::Update released = hold(gesture, false);
    CHECK(released.click);
    CHECK_FALSE(released.look_active);
    // Never entered look, so the cursor was never grabbed and must not be
    // released -- doing so would show a cursor that was never hidden.
    CHECK_FALSE(released.began_look);
    CHECK_FALSE(released.ended_look);
}

TEST_CASE("the click reports the press position, not the release position", "[drag_gesture]") {
    DragGesture gesture;

    hold(gesture, true, 640.0, 360.0);
    // Within the threshold, so still a click -- releasing a physical button
    // nudges the cursor a pixel or two.
    const DragGesture::Update released = gesture.update(false, 642.0, 361.0);

    REQUIRE(released.click);
    CHECK(released.click_x == 640.0);
    CHECK(released.click_y == 360.0);
}

TEST_CASE("movement past the threshold becomes a camera drag", "[drag_gesture]") {
    DragGesture gesture;

    hold(gesture, true, 100.0, 100.0);

    // Just under the threshold: still undecided, and still a click if released.
    const DragGesture::Update nudged = gesture.update(true, 103.0, 100.0);
    CHECK_FALSE(nudged.look_active);
    CHECK_FALSE(nudged.began_look);

    const DragGesture::Update dragged = gesture.update(true, 120.0, 100.0);
    CHECK(dragged.look_active);
    CHECK(dragged.began_look);

    const DragGesture::Update released = gesture.update(false, 120.0, 100.0);
    CHECK(released.ended_look);
    // The whole point: a drag must not also open a menu.
    CHECK_FALSE(released.click);
    CHECK_FALSE(released.look_active);
}

TEST_CASE("the threshold is a radius, not an axis", "[drag_gesture]") {
    DragGesture gesture;
    hold(gesture, true, 0.0, 0.0);

    // 3,3 is 4.24 away -- under the threshold on either axis alone, over it
    // diagonally. Testing each axis separately would miss this.
    const DragGesture::Update diagonal = gesture.update(true, 3.0, 3.0);
    CHECK(diagonal.look_active);
}

TEST_CASE("look begins once and ends once", "[drag_gesture]") {
    DragGesture gesture;

    hold(gesture, true, 0.0, 0.0);
    CHECK(gesture.update(true, 50.0, 0.0).began_look);
    // Still dragging: the grab already happened and must not be repeated.
    CHECK_FALSE(gesture.update(true, 90.0, 0.0).began_look);
    CHECK_FALSE(gesture.update(true, 130.0, 0.0).began_look);

    CHECK(gesture.update(false, 130.0, 0.0).ended_look);
    // Released again with nothing held: no transition, and no stray click.
    const DragGesture::Update idle = gesture.update(false, 130.0, 0.0);
    CHECK_FALSE(idle.ended_look);
    CHECK_FALSE(idle.click);
}

TEST_CASE("a second gesture works after the first", "[drag_gesture]") {
    DragGesture gesture;

    hold(gesture, true);
    CHECK(hold(gesture, false).click);

    // A drag after a click must not inherit the click's press position or
    // leave the gesture stuck undecided.
    gesture.update(true, 200.0, 200.0);
    CHECK(gesture.update(true, 260.0, 200.0).began_look);
    CHECK(gesture.update(false, 260.0, 200.0).ended_look);

    // And a click after a drag is still a click.
    hold(gesture, true, 300.0, 300.0);
    const DragGesture::Update second = hold(gesture, false, 300.0, 300.0);
    CHECK(second.click);
    CHECK(second.click_x == 300.0);
}

TEST_CASE("a release with no press is nothing at all", "[drag_gesture]") {
    DragGesture gesture;

    const DragGesture::Update idle = hold(gesture, false);
    CHECK_FALSE(idle.click);
    CHECK_FALSE(idle.look_active);
    CHECK_FALSE(idle.began_look);
    CHECK_FALSE(idle.ended_look);
}
