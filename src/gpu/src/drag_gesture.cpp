#include <sage/gpu/drag_gesture.hpp>

namespace sage::gpu {

DragGesture::Update DragGesture::update(bool button_down, double cursor_x, double cursor_y) {
    Update result;

    if (button_down && !active_ && !pending_) {
        pending_ = true;
        press_x_ = cursor_x;
        press_y_ = cursor_y;
    }

    if (pending_ && button_down) {
        const double dx = cursor_x - press_x_;
        const double dy = cursor_y - press_y_;
        if ((dx * dx) + (dy * dy) >= k_drag_threshold * k_drag_threshold) {
            pending_ = false;
            active_ = true;
            result.began_look = true;
        }
    }

    if (!button_down) {
        if (pending_) {
            pending_ = false;
            result.click = true;
            result.click_x = press_x_;
            result.click_y = press_y_;
        }
        if (active_) {
            active_ = false;
            result.ended_look = true;
        }
    }

    result.look_active = active_;
    return result;
}

}  // namespace sage::gpu
