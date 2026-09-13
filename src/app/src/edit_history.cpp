#include <sage/app/edit_history.hpp>
#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>

#include <utility>

namespace sage::app {

EditHistory::EditHistory(std::size_t max_depth) : max_depth_(max_depth) {
    SAGE_VERIFY(max_depth > 0, "EditHistory: max_depth must be at least one");
}

void EditHistory::push(std::string name, std::function<void()> undo_action,
                       std::function<void()> redo_action) {
    // Anything already undone is dropped; see the class comment for why the
    // history is a line rather than a tree.
    commands_.resize(cursor_);
    commands_.push_back(Command{std::move(name), std::move(undo_action), std::move(redo_action)});

    // Only ever one over the cap, since this is the one place that grows the
    // vector. Dropping from the front costs an O(n) shift, which happens once
    // per edit past the cap and is nothing next to the edit itself.
    if (commands_.size() > max_depth_) {
        commands_.erase(commands_.begin());
    }
    cursor_ = commands_.size();
}

bool EditHistory::undo() {
    if (!can_undo()) {
        return false;
    }
    --cursor_;
    commands_[cursor_].undo();
    SAGE_LOG_INFO("Undo: {}", commands_[cursor_].name);
    return true;
}

bool EditHistory::redo() {
    if (!can_redo()) {
        return false;
    }
    commands_[cursor_].redo();
    SAGE_LOG_INFO("Redo: {}", commands_[cursor_].name);
    ++cursor_;
    return true;
}

void EditHistory::clear() {
    commands_.clear();
    cursor_ = 0;
}

std::string_view EditHistory::undo_name() const {
    return can_undo() ? std::string_view{commands_[cursor_ - 1].name} : std::string_view{};
}

std::string_view EditHistory::redo_name() const {
    return can_redo() ? std::string_view{commands_[cursor_].name} : std::string_view{};
}

}  // namespace sage::app
