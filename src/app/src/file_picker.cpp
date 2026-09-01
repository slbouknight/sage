#include "file_picker.hpp"

#include <sage/core/log.hpp>

#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <system_error>
#include <utility>

namespace sage::app {

namespace {

constexpr float k_list_height = 240.0F;

bool is_gltf(const std::filesystem::path& path) {
    // glTF is case-insensitive in practice; a file written as .GLB on a
    // case-preserving filesystem is still a glTF.
    std::string extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".gltf" || extension == ".glb";
}

}  // namespace

FilePicker::FilePicker(const std::filesystem::path& start_directory) {
    navigate_to(start_directory);

    // A bare filename argument has no parent path, and a stale one may not
    // exist. Either way the picker must open somewhere rather than nowhere.
    if (directory_.empty()) {
        std::error_code error;
        navigate_to(std::filesystem::current_path(error));
    }
}

void FilePicker::navigate_to(const std::filesystem::path& directory) {
    std::error_code error;
    std::filesystem::path resolved = std::filesystem::weakly_canonical(directory, error);
    if (error || !std::filesystem::is_directory(resolved, error)) {
        status_ = "Not a directory: " + directory.string();
        return;
    }

    directory_ = std::move(resolved);
    // The text field follows the tree, so typing a path and browsing to one
    // leave the field saying the same thing.
    path_input_ = directory_.string();
    refresh();
}

void FilePicker::refresh() {
    entries_.clear();
    selected_ = -1;
    status_.clear();

    if (directory_.has_parent_path() && directory_.parent_path() != directory_) {
        entries_.push_back({"..", directory_.parent_path(), true});
    }

    std::error_code error;
    // The non-throwing overload plus skip_permission_denied: a browsable tree
    // routinely contains directories this process cannot open, and one of them
    // must not end the listing.
    const std::filesystem::directory_iterator end;
    std::filesystem::directory_iterator it(
        directory_, std::filesystem::directory_options::skip_permission_denied, error);
    if (error) {
        status_ = "Cannot read directory: " + error.message();
        return;
    }

    std::vector<Entry> directories;
    std::vector<Entry> files;
    for (; it != end; it.increment(error)) {
        if (error) {
            // A single unreadable entry ends iteration if left unhandled.
            // Clearing it and stopping keeps whatever was already listed.
            status_ = "Listing stopped early: " + error.message();
            break;
        }

        const std::filesystem::path& path = it->path();
        const std::string name = path.filename().string();
        if (name.starts_with('.')) {
            continue;
        }

        std::error_code kind_error;
        if (it->is_directory(kind_error)) {
            directories.push_back({name + "/", path, true});
        } else if (is_gltf(path)) {
            files.push_back({name, path, false});
        }
    }

    const auto by_label = [](const Entry& a, const Entry& b) { return a.label < b.label; };
    std::ranges::sort(directories, by_label);
    std::ranges::sort(files, by_label);

    // Directories first, so the navigation controls stay together at the top
    // rather than interleaving with the files they contain.
    entries_.insert(entries_.end(), directories.begin(), directories.end());
    entries_.insert(entries_.end(), files.begin(), files.end());
}

std::optional<FilePicker::Request> FilePicker::draw() {
    clear_requested_ = false;
    std::optional<Request> request;

    ImGui::Begin("Load glTF");

    // Chosen on this frame, resolved after the list so a navigation does not
    // invalidate the entry being read out from under the loop.
    std::optional<std::filesystem::path> chosen_directory;

    ImGui::TextWrapped("%s", directory_.string().c_str());
    if (ImGui::Button("Refresh")) {
        refresh();
    }
    ImGui::SameLine();
    ImGui::Checkbox("Replace scene", &replace_);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        clear_requested_ = true;
    }

    ImGui::BeginChild("entries", {0.0F, k_list_height}, ImGuiChildFlags_Borders);
    for (int i = 0; i < static_cast<int>(entries_.size()); ++i) {
        const Entry& entry = entries_[static_cast<std::size_t>(i)];

        ImGui::PushID(i);
        // AllowDoubleClick still reports the first click, so single-click
        // selects and double-click acts -- the two are not exclusive.
        if (ImGui::Selectable(entry.label.c_str(), selected_ == i,
                              ImGuiSelectableFlags_AllowDoubleClick)) {
            selected_ = i;
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                if (entry.is_directory) {
                    chosen_directory = entry.path;
                } else {
                    request = Request{entry.path, replace_};
                }
            }
        }
        ImGui::PopID();
    }
    ImGui::EndChild();

    // Enter in the field acts, so a pasted path does not also need a click.
    const bool submitted =
        ImGui::InputText("##path", &path_input_, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    const bool pressed = ImGui::Button("Load");

    if (submitted || pressed) {
        const std::filesystem::path typed(path_input_);
        std::error_code error;
        if (std::filesystem::is_directory(typed, error)) {
            chosen_directory = typed;
        } else if (is_gltf(typed)) {
            // Existence is not checked here; the loader reports a bad path with
            // the parser's own message, which says more than "not found".
            request = Request{typed, replace_};
        } else if (selected_ >= 0 && !entries_[static_cast<std::size_t>(selected_)].is_directory) {
            // The field was left on a directory, so fall back to the selection.
            request = Request{entries_[static_cast<std::size_t>(selected_)].path, replace_};
        } else {
            status_ = "Not a glTF file or directory: " + path_input_;
        }
    }

    if (!status_.empty()) {
        ImGui::TextWrapped("%s", status_.c_str());
    }

    ImGui::End();

    if (chosen_directory.has_value()) {
        navigate_to(*chosen_directory);
    }

    return request;
}

}  // namespace sage::app
