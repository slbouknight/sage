#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace sage::app {

// A directory browser for glTF files, drawn with ImGui over std::filesystem.
//
// Not a native dialog: on Linux that means an xdg-desktop-portal round trip or
// a toolkit dependency, and this has one job -- point at a .gltf or .glb. The
// text field is the escape hatch for anywhere the tree is tedious to reach.
class FilePicker {
public:
    // What the picker asks the application to do. Returned by value rather than
    // acted on here, because loading is not something a UI callback should do
    // mid-frame; see Application::service_pending_load.
    struct Request {
        std::filesystem::path path;
        // False adds to the current scene instead of replacing it. Both work:
        // the registries bump-allocate and only rewind on a clear.
        bool replace = true;
    };

    explicit FilePicker(const std::filesystem::path& start_directory);

    // Returns a request on the frame a file is chosen, nothing otherwise.
    [[nodiscard]] std::optional<Request> draw();

    // True on the frame the user asks for an empty scene.
    [[nodiscard]] bool clear_requested() const { return clear_requested_; }

private:
    struct Entry {
        std::string label;
        std::filesystem::path path;
        bool is_directory = false;
    };

    void navigate_to(const std::filesystem::path& directory);
    void refresh();

    std::filesystem::path directory_;
    std::vector<Entry> entries_;
    // Index into entries_; -1 when nothing is selected, which is why it is
    // signed. Reset by every navigation, since the index would otherwise name
    // an unrelated file in the new listing.
    int selected_ = -1;
    bool replace_ = true;
    bool clear_requested_ = false;
    std::string path_input_;
    // Surfaced in the panel rather than only the log, because a directory that
    // cannot be read is something the person clicking needs to see.
    std::string status_;
};

}  // namespace sage::app
