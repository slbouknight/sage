#include <sage/app/edit_history.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using sage::app::EditHistory;

namespace {

// The closures are the only observable effect a command has, so every test
// here works by appending to a log and asserting on what ran, in what order.
// A command named "a" appends "a-undo" or "a-redo".
struct Recorder {
    std::vector<std::string> log;

    void push_to(EditHistory& history, std::string name) {
        const std::string undo_mark = name + "-undo";
        const std::string redo_mark = name + "-redo";
        history.push(
            std::move(name), [this, undo_mark]() { log.push_back(undo_mark); },
            [this, redo_mark]() { log.push_back(redo_mark); });
    }
};

}  // namespace

TEST_CASE("a fresh history has nothing to undo or redo", "[edit_history]") {
    const EditHistory history;

    CHECK_FALSE(history.can_undo());
    CHECK_FALSE(history.can_redo());
    CHECK(history.size() == 0);
    CHECK(history.cursor() == 0);
    CHECK(history.undo_name().empty());
    CHECK(history.redo_name().empty());
}

TEST_CASE("undo and redo on an empty history are no-ops, not underflows", "[edit_history]") {
    EditHistory history;

    // cursor_ is unsigned: a missing guard here would not fail loudly, it
    // would wrap to SIZE_MAX and index the vector out of bounds on the next
    // undo. This is the single most important thing in the file.
    CHECK_FALSE(history.undo());
    CHECK_FALSE(history.redo());
    CHECK(history.cursor() == 0);
    CHECK(history.size() == 0);
}

TEST_CASE("push records without applying", "[edit_history]") {
    EditHistory history;
    Recorder recorder;

    recorder.push_to(history, "a");

    // The caller already performed the edit; push only remembers how to
    // reverse it. Running redo here would apply it a second time.
    CHECK(recorder.log.empty());
    CHECK(history.can_undo());
    CHECK_FALSE(history.can_redo());
    CHECK(history.size() == 1);
    CHECK(history.cursor() == 1);
}

TEST_CASE("undo runs the undo closure and steps the cursor back", "[edit_history]") {
    EditHistory history;
    Recorder recorder;
    recorder.push_to(history, "a");

    CHECK(history.undo());

    CHECK(recorder.log == std::vector<std::string>{"a-undo"});
    CHECK(history.cursor() == 0);
    // Undone, not discarded: the command is still there to redo.
    CHECK(history.size() == 1);
    CHECK_FALSE(history.can_undo());
    CHECK(history.can_redo());
}

TEST_CASE("redo runs the redo closure and steps the cursor forward", "[edit_history]") {
    EditHistory history;
    Recorder recorder;
    recorder.push_to(history, "a");
    history.undo();
    recorder.log.clear();

    CHECK(history.redo());

    CHECK(recorder.log == std::vector<std::string>{"a-redo"});
    CHECK(history.cursor() == 1);
    CHECK(history.can_undo());
    CHECK_FALSE(history.can_redo());
}

TEST_CASE("undo unwinds in reverse order", "[edit_history]") {
    EditHistory history;
    Recorder recorder;
    recorder.push_to(history, "a");
    recorder.push_to(history, "b");
    recorder.push_to(history, "c");

    history.undo();
    history.undo();
    history.undo();

    CHECK(recorder.log == std::vector<std::string>{"c-undo", "b-undo", "a-undo"});
    CHECK(history.cursor() == 0);
    CHECK_FALSE(history.undo());
}

TEST_CASE("redo replays in forward order", "[edit_history]") {
    EditHistory history;
    Recorder recorder;
    recorder.push_to(history, "a");
    recorder.push_to(history, "b");
    history.undo();
    history.undo();
    recorder.log.clear();

    history.redo();
    history.redo();

    CHECK(recorder.log == std::vector<std::string>{"a-redo", "b-redo"});
    CHECK(history.cursor() == 2);
    CHECK_FALSE(history.redo());
}

TEST_CASE("a new edit after an undo discards the redo branch", "[edit_history]") {
    EditHistory history;
    Recorder recorder;
    recorder.push_to(history, "a");
    recorder.push_to(history, "b");
    recorder.push_to(history, "c");

    history.undo();  // c undone
    history.undo();  // b undone
    recorder.log.clear();

    recorder.push_to(history, "d");

    // b and c are gone: the history is a line, so d replaces what was ahead.
    CHECK(history.size() == 2);
    CHECK(history.cursor() == 2);
    CHECK_FALSE(history.can_redo());

    // And what remains really is a-then-d, not a-then-b.
    history.undo();
    history.undo();
    CHECK(recorder.log == std::vector<std::string>{"d-undo", "a-undo"});
}

TEST_CASE("the discarded branch's closures never run again", "[edit_history]") {
    EditHistory history;
    Recorder recorder;
    recorder.push_to(history, "doomed");
    history.undo();
    recorder.log.clear();

    recorder.push_to(history, "replacement");
    // Redo must not resurrect the abandoned command.
    CHECK_FALSE(history.redo());
    CHECK(recorder.log.empty());
}

TEST_CASE("names label what undo and redo would do", "[edit_history]") {
    EditHistory history;
    Recorder recorder;
    recorder.push_to(history, "Transform");
    recorder.push_to(history, "Delete");

    CHECK(history.undo_name() == "Delete");
    CHECK(history.redo_name().empty());

    history.undo();
    CHECK(history.undo_name() == "Transform");
    CHECK(history.redo_name() == "Delete");

    history.undo();
    CHECK(history.undo_name().empty());
    CHECK(history.redo_name() == "Transform");
}

TEST_CASE("clear drops everything in both directions", "[edit_history]") {
    EditHistory history;
    Recorder recorder;
    recorder.push_to(history, "a");
    recorder.push_to(history, "b");
    history.undo();
    recorder.log.clear();

    history.clear();

    CHECK(history.size() == 0);
    CHECK(history.cursor() == 0);
    CHECK_FALSE(history.can_undo());
    CHECK_FALSE(history.can_redo());
    CHECK_FALSE(history.undo());
    CHECK_FALSE(history.redo());
    CHECK(recorder.log.empty());
}

TEST_CASE("the stack is capped, dropping the oldest command", "[edit_history]") {
    EditHistory history{3};
    Recorder recorder;

    recorder.push_to(history, "a");
    recorder.push_to(history, "b");
    recorder.push_to(history, "c");
    CHECK(history.size() == 3);

    recorder.push_to(history, "d");

    // Still three, and the cursor still sits at the end -- a trim must not
    // leave the cursor pointing past what survived.
    CHECK(history.size() == 3);
    CHECK(history.cursor() == 3);

    // "a" is what fell off the front.
    history.undo();
    history.undo();
    history.undo();
    CHECK(recorder.log == std::vector<std::string>{"d-undo", "c-undo", "b-undo"});
    CHECK_FALSE(history.can_undo());
}

TEST_CASE("a full stack stays usable across undo and further edits", "[edit_history]") {
    EditHistory history{2};
    Recorder recorder;
    recorder.push_to(history, "a");
    recorder.push_to(history, "b");
    history.undo();  // b undone, cursor 1

    // The truncation to the cursor happens before the cap is tested, so this
    // replaces "b" rather than pushing the stack over the limit and evicting
    // "a" as well.
    recorder.push_to(history, "c");
    CHECK(history.size() == 2);
    CHECK(history.cursor() == 2);

    recorder.log.clear();
    history.undo();
    history.undo();
    CHECK(recorder.log == std::vector<std::string>{"c-undo", "a-undo"});
}

TEST_CASE("a depth-one history keeps only the latest edit", "[edit_history]") {
    EditHistory history{1};
    Recorder recorder;
    recorder.push_to(history, "a");
    recorder.push_to(history, "b");

    CHECK(history.size() == 1);
    CHECK(history.cursor() == 1);

    history.undo();
    CHECK(recorder.log == std::vector<std::string>{"b-undo"});
    CHECK_FALSE(history.can_undo());
}

TEST_CASE("undo and redo interleave without drifting", "[edit_history]") {
    EditHistory history;
    Recorder recorder;
    recorder.push_to(history, "a");
    recorder.push_to(history, "b");

    for (int i = 0; i < 5; ++i) {
        history.undo();
        history.redo();
    }

    CHECK(history.cursor() == 2);
    CHECK(history.size() == 2);
    // Two entries per iteration -- one undo, one redo -- over five iterations.
    CHECK(recorder.log.size() == 10);
    CHECK(recorder.log.front() == "b-undo");
    CHECK(recorder.log.back() == "b-redo");
}
