#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace sage::app {

// A linear undo/redo history over closures.
//
// Closures rather than a variant of command types: every operation the editor
// records is a handful of captured values and two calls into the scene graph,
// and a class hierarchy for that would be more machinery than the thing it
// describes.
//
// Deliberately knows nothing about the scene, Vulkan or ImGui. What a command
// *does* is entirely inside the two functions handed to push(); this owns only
// the ordering, the cursor and the cap. That is what makes it testable without
// a device, which is the reason it is a type of its own rather than four
// members on Application.
class EditHistory {
public:
    // `max_depth` bounds the stack so a long session cannot grow it without
    // limit. Once full, the oldest command is dropped -- so the very first
    // edits of a session eventually stop being reachable, which is the usual
    // bargain and is why the default is generous.
    explicit EditHistory(std::size_t max_depth = k_default_max_depth);

    // Records an edit that has *already been applied*. push() does not call
    // `redo` -- the caller performed the action, this only remembers how to
    // reverse it.
    //
    // Anything currently undone is discarded: the history is a line, not a
    // tree, so an edit made after stepping back replaces what was ahead.
    void push(std::string name, std::function<void()> undo, std::function<void()> redo);

    // Steps back one command and runs its undo. False when there was nothing
    // to undo, which callers may ignore -- it is not an error, just a no-op.
    bool undo();
    // Runs the next command's redo and steps forward. False when the cursor is
    // already at the end.
    bool redo();

    // Drops every command. For edits that genuinely cannot be reversed -- a
    // registry rewind -- where leaving the history in place would offer an
    // undo that could not be honoured.
    void clear();

    [[nodiscard]] bool can_undo() const { return cursor_ > 0; }
    [[nodiscard]] bool can_redo() const { return cursor_ < commands_.size(); }

    // What undo/redo would reverse or reapply, for labelling a menu item.
    // Empty when the corresponding direction is unavailable. The view is
    // invalidated by any mutation, so use it and let it go.
    [[nodiscard]] std::string_view undo_name() const;
    [[nodiscard]] std::string_view redo_name() const;

    // Commands retained, undone ones included. Distinct from the cursor, and
    // both are exposed because the distinction is exactly what the tests pin.
    [[nodiscard]] std::size_t size() const { return commands_.size(); }
    // How many commands are currently applied. Everything at or past it has
    // been undone.
    [[nodiscard]] std::size_t cursor() const { return cursor_; }
    [[nodiscard]] std::size_t max_depth() const { return max_depth_; }

    static constexpr std::size_t k_default_max_depth = 128;

private:
    struct Command {
        std::string name;
        std::function<void()> undo;
        std::function<void()> redo;
    };

    std::vector<Command> commands_;
    std::size_t cursor_ = 0;
    std::size_t max_depth_;
};

}  // namespace sage::app
