#pragma once

namespace sage::gpu {

// Decides what a right-button press turned out to mean.
//
// The button does two jobs: held and dragged it flies the camera, clicked and
// released it opens a context menu. Which one it was is not knowable at press
// time, so this holds the press in an undecided state until the cursor either
// moves far enough to be a drag or the button comes back up.
//
// Split out of Window and kept free of GLFW so the decision can be tested
// without a window and a human to click in it -- there is no way to drive a
// real cursor from a test, and this is the part that would be wrong.
class DragGesture {
public:
    // How far the cursor must travel, in pixels, before a press counts as a
    // drag. Comfortably above the jitter of clicking a physical button, and
    // well below anything anyone would perform deliberately as a drag.
    static constexpr double k_drag_threshold = 4.0;

    struct Update {
        // Currently flying the camera.
        bool look_active = false;
        // Transitions, which is when the cursor has to be grabbed or released.
        // Reported separately because doing it every frame would fight the
        // window system.
        bool began_look = false;
        bool ended_look = false;
        // Released without ever having become a drag.
        bool click = false;
        // Where the press happened, which is the anchor a menu opens at. The
        // release position is within the threshold of it, but the press is
        // where the user was actually pointing.
        double click_x = 0.0;
        double click_y = 0.0;
    };

    // Call once per frame with the button state and the cursor position in
    // window pixels. While look_active, the caller is expected to have grabbed
    // the cursor, so the position it passes is no longer a screen coordinate --
    // this deliberately stops reading it once the drag is committed.
    Update update(bool button_down, double cursor_x, double cursor_y);

private:
    bool active_ = false;
    bool pending_ = false;
    double press_x_ = 0.0;
    double press_y_ = 0.0;
};

}  // namespace sage::gpu
