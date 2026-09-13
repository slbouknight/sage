#include <sage/app/scene_query.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/imgui_layer.hpp>
#include <sage/gpu/light.hpp>
#include <sage/gpu/primitives.hpp>

#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

// After imgui.h: both of these use its types without including it. The
// DockBuilder API imgui_internal.h carries is internal and has no public
// equivalent; it is used here only for the one-time default layout, and
// everything else goes through the public header.
#include <ImGuizmo.h>
#include <imgui_internal.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "application.hpp"

// The editor's panels: the dockspace, the menu bar, the overlay, the two docked
// panels, the context menu and the gizmo.
//
// Application's members rather than a class of their own. These read and write
// the scene, the selection, the undo history and the pending queues -- they are
// the editor's UI, not a component with an interface of its own, and wrapping
// them in a type holding an Application& would be the same coupling one
// indirection deeper plus a header to keep in step.
//
// What the split does buy: application.cpp stops being two unrelated files
// stapled together, ImGui and ImGuizmo stop being included by the translation
// unit that holds the frame loop and the scene editing, and a change to a panel
// no longer rebuilds either.
//
// The genuinely separable part left with scene_query: build_child_table, which
// is index bookkeeping over tombstoned nodes and is tested there.

namespace sage::app {

namespace {

// The dockspace's central node in framebuffer pixels, or the whole image when
// there is no layout yet. Free rather than a member so ImGuiID stays out of
// application.hpp.
VkRect2D central_node_rect(ImGuiID dockspace, VkExtent2D extent) {
    const VkRect2D whole{{0, 0}, extent};

    const ImGuiDockNode* central = ImGui::DockBuilderGetCentralNode(dockspace);
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (central == nullptr || central->Size.x <= 0.0F || central->Size.y <= 0.0F ||
        viewport->Size.x <= 0.0F || viewport->Size.y <= 0.0F) {
        return whole;
    }

    // ImGui measures in the window's logical coordinates while the swapchain is
    // in framebuffer pixels. The two are equal until display scaling is
    // involved -- exactly the kind of difference that goes unnoticed on one
    // machine and misplaces the whole viewport on another.
    const float scale_x = static_cast<float>(extent.width) / viewport->Size.x;
    const float scale_y = static_cast<float>(extent.height) / viewport->Size.y;

    // Relative to the viewport rather than absolute: with multi-viewport off
    // the main viewport sits at the origin, but subtracting costs nothing and
    // is correct either way.
    const float x = (central->Pos.x - viewport->Pos.x) * scale_x;
    const float y = (central->Pos.y - viewport->Pos.y) * scale_y;

    // Clamped because a scissor rect reaching outside the attachment is
    // invalid, and rounding at the edges is enough to put it there.
    const auto clamp_to = [](float value, std::uint32_t limit) {
        return static_cast<std::uint32_t>(std::clamp(value, 0.0F, static_cast<float>(limit)));
    };
    const std::uint32_t left = clamp_to(x, extent.width);
    const std::uint32_t top = clamp_to(y, extent.height);

    VkRect2D rect{};
    rect.offset = {static_cast<std::int32_t>(left), static_cast<std::int32_t>(top)};
    rect.extent = {clamp_to(central->Size.x * scale_x, extent.width - left),
                   clamp_to(central->Size.y * scale_y, extent.height - top)};

    // A degenerate node -- collapsed, or mid-resize -- would divide by zero in
    // the projection and make an invalid viewport.
    if (rect.extent.width == 0 || rect.extent.height == 0) {
        return whole;
    }
    return rect;
}

// Range of the exposure slider, in stops. Four either way covers "the scene is
// lit for a sunny day" to "the scene is lit by one lantern" without the slider
// becoming too coarse to make a small correction with.
constexpr float k_min_exposure_stops = -4.0F;
constexpr float k_max_exposure_stops = 4.0F;

// The one docked column, as a fraction of the window. It carries the scene
// tree over the selection's properties; everything else is a menu, an overlay
// or a dialog, and the rest of the window is the 3D view.
constexpr float k_right_column_fraction = 0.22F;
// Inset of the read-out from the 3D view's top-right corner, and how opaque
// its backing is: enough to stay legible over a bright surface without hiding
// what is behind it.
constexpr float k_overlay_margin = 12.0F;
constexpr float k_overlay_alpha = 0.55F;

// Menu content has no panel to stretch into, so widgets that would otherwise
// fill the available width need one given to them.
constexpr float k_menu_item_width = 220.0F;
// The right column is split between the scene tree and the properties of
// whatever is selected in it.
constexpr float k_hierarchy_fraction = 0.5F;

}  // namespace

void Application::draw_dockspace() {
    // With the UI hidden there is no dockspace at all, and the 3D view takes
    // the whole image. Set explicitly rather than by letting the central node
    // grow: a node whose windows were simply not submitted keeps whatever size
    // the layout gave it, so the rect would be a frame behind at best.
    if (ui_hidden_) {
        viewport_rect_ = VkRect2D{{0, 0}, swapchain_.extent()};
        const ImGuiViewport* whole = ImGui::GetMainViewport();
        viewport_logical_pos_ = glm::vec2(whole->Pos.x, whole->Pos.y);
        viewport_logical_size_ = glm::vec2(whole->Size.x, whole->Size.y);
        return;
    }

    // PassthruCentralNode leaves the middle node transparent and, while it is
    // empty, lets mouse input through it -- so the scene shows and the camera
    // still responds. NoDockingOverCentralNode keeps it empty permanently, so a
    // panel cannot be dragged over the 3D view by accident.
    const ImGuiID dockspace = ImGui::DockSpaceOverViewport(
        0, ImGui::GetMainViewport(),
        ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_NoDockingOverCentralNode);

    viewport_rect_ = central_node_rect(dockspace, swapchain_.extent());
    update_viewport_logical_rect(dockspace);

    if (dock_layout_built_) {
        return;
    }
    dock_layout_built_ = true;

    // Rebuilt from scratch every run. Layout persistence would be io.IniFilename,
    // which ImGuiLayer deliberately leaves null, so there is nothing to preserve
    // and the arrangement is the same on every start.
    ImGui::DockBuilderRemoveNode(dockspace);
    ImGui::DockBuilderAddNode(dockspace, ImGuiDockNodeFlags_DockSpace);
    // Set before splitting: the ratios below are fractions of the node's size,
    // and a node that has not been sized yet splits unpredictably.
    ImGui::DockBuilderSetNodeSize(dockspace, ImGui::GetMainViewport()->Size);

    // One column, on the right. The left one is gone: the read-out it held is
    // now an overlay inside the 3D view and the browser it held is a dialog off
    // the File menu, so a whole column of screen was being spent on two things
    // that needed no permanent home.
    ImGuiID right = 0;
    ImGuiID centre = 0;
    ImGui::DockBuilderSplitNode(dockspace, ImGuiDir_Right, k_right_column_fraction, &right,
                                &centre);

    ImGuiID right_top = 0;
    ImGuiID right_bottom = 0;
    ImGui::DockBuilderSplitNode(right, ImGuiDir_Up, k_hierarchy_fraction, &right_top,
                                &right_bottom);

    ImGui::DockBuilderDockWindow("Hierarchy", right_top);
    ImGui::DockBuilderDockWindow("Properties", right_bottom);
    ImGui::DockBuilderFinish(dockspace);

    // The split above changed the central node, so the rect taken before it is
    // stale for this frame.
    viewport_rect_ = central_node_rect(dockspace, swapchain_.extent());
    update_viewport_logical_rect(dockspace);
}

void Application::update_viewport_logical_rect(unsigned int dockspace) {
    const ImGuiDockNode* central = ImGui::DockBuilderGetCentralNode(dockspace);
    const ImGuiViewport* whole = ImGui::GetMainViewport();
    if (central == nullptr || central->Size.x <= 0.0F || central->Size.y <= 0.0F) {
        viewport_logical_pos_ = glm::vec2(whole->Pos.x, whole->Pos.y);
        viewport_logical_size_ = glm::vec2(whole->Size.x, whole->Size.y);
        return;
    }
    viewport_logical_pos_ = glm::vec2(central->Pos.x, central->Pos.y);
    viewport_logical_size_ = glm::vec2(central->Size.x, central->Size.y);
}

void Application::draw_stats_overlay() {
    // Pinned inside the 3D view rather than to the window, so it tracks the
    // central node as panels resize and follows the whole screen once they are
    // hidden.
    const ImVec2 corner{viewport_logical_pos_.x + viewport_logical_size_.x - k_overlay_margin,
                        viewport_logical_pos_.y + k_overlay_margin};
    ImGui::SetNextWindowPos(corner, ImGuiCond_Always, ImVec2(1.0F, 0.0F));
    ImGui::SetNextWindowBgAlpha(k_overlay_alpha);

    // NoInputs is the one that matters: without it the overlay would swallow
    // clicks meant for whatever is behind it, and picking would go dead in one
    // corner of the viewport for no visible reason.
    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoInputs;

    ImGui::Begin("##stats", nullptr, flags);
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("%.1f fps  %.2f ms", static_cast<double>(io.Framerate),
                1000.0 / static_cast<double>(io.Framerate));
    ImGui::Separator();
    ImGui::Text("%zu nodes", scene_graph_.live_size());
    ImGui::Text("Geometry  %llu / %llu KiB",
                static_cast<unsigned long long>(geometry_registry_.used() / 1024),
                static_cast<unsigned long long>(geometry_registry_.capacity() / 1024));
    ImGui::Text("Materials %u / %u", material_registry_.count(), material_registry_.capacity());
    ImGui::Text("Textures  %u", texture_registry_.count());
    ImGui::Separator();
    const gpu::SceneNode* selected = scene_graph_.find(selected_);
    ImGui::Text("Selected  %s", selected != nullptr ? selected->name.c_str() : "(none)");
    const glm::vec3 position = camera_.position();
    ImGui::Text("Camera    %.1f, %.1f, %.1f", static_cast<double>(position.x),
                static_cast<double>(position.y), static_cast<double>(position.z));
    ImGui::End();
}

void Application::draw_menu_bar() {
    if (!ImGui::BeginMainMenuBar()) {
        return;
    }

    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open glTF...")) {
            file_dialog_open_ = true;
            file_dialog_scene_actions_ = true;
            // Absent, so the file lands where it says it does. Only the context
            // menu places a load.
            file_dialog_placement_.reset();
        }
        if (ImGui::MenuItem("Clear scene")) {
            // Queued: clear_scene waits for the device to go idle and destroys
            // images a recording command buffer still names.
            pending_clear_ = true;
        }
        ImGui::Separator();
        ImGui::MenuItem("Include UI in captures", nullptr, &screenshot_include_ui_);
        if (ImGui::MenuItem("Screenshot", "F2", false, !pending_screenshot_.has_value())) {
            request_screenshot();
        }
        // What the next capture will actually contain, which the toggle above
        // decides and is otherwise only discoverable by taking one.
        if (screenshot_include_ui_) {
            ImGui::TextDisabled("%ux%u, whole window", swapchain_.extent().width,
                                swapchain_.extent().height);
        } else {
            ImGui::TextDisabled("%ux%u, no UI", viewport_rect_.extent.width,
                                viewport_rect_.extent.height);
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Edit")) {
        // Labelled with what they would actually reverse, so the menu says
        // "Undo Transform" rather than leaving you to remember what you did.
        const std::string undo_label =
            history_.can_undo() ? "Undo " + std::string(history_.undo_name()) : std::string("Undo");
        const std::string redo_label =
            history_.can_redo() ? "Redo " + std::string(history_.redo_name()) : std::string("Redo");
        if (ImGui::MenuItem(undo_label.c_str(), "Ctrl+Z", false, history_.can_undo())) {
            history_.undo();
        }
        if (ImGui::MenuItem(redo_label.c_str(), "Ctrl+Y", false, history_.can_redo())) {
            history_.redo();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Delete", "Del", false, scene_graph_.find(selected_) != nullptr)) {
            delete_selected();
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Render")) {
        draw_presentation_controls();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Lighting")) {
        draw_lighting_menu();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem("Hide panels", "F11", &ui_hidden_);
        ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
}

void Application::draw_file_dialog() {
    if (file_dialog_open_ && !ImGui::IsPopupOpen("Load glTF")) {
        ImGui::OpenPopup("Load glTF");
    }

    // Centred rather than at the cursor: the browser is a good deal larger than
    // a menu, and anchoring it to a click near an edge would push it off-screen.
    const ImVec2 centre = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(centre, ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));
    if (!ImGui::BeginPopupModal("Load glTF", &file_dialog_open_,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }

    if (const std::optional<FilePicker::Request> request =
            file_picker_.draw_contents(file_dialog_scene_actions_);
        request.has_value()) {
        // Queued rather than loaded here: this is the middle of a frame, and a
        // load blocks, waits for the device and destroys images the command
        // buffer being recorded would still reference.
        pending_load_ = PendingLoad{request->path, request->replace, file_dialog_placement_};
        file_dialog_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    if (file_picker_.clear_requested()) {
        pending_clear_ = true;
        file_dialog_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    if (ImGui::Button("Cancel")) {
        file_dialog_open_ = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void Application::draw_presentation_controls() {
    ImGui::SeparatorText("Tonemap");

    // Selectable at runtime rather than baked in, because the difference between
    // two curves is only legible on the same frame -- comparing across a rebuild
    // compares two memories of an image.
    ImGui::SetNextItemWidth(k_menu_item_width);
    ImGui::Combo("##tonemap", &render_settings_.tonemap_operator, k_tonemap_names.data(),
                 static_cast<int>(k_tonemap_names.size()));
    if (render_settings_.tonemap_operator == static_cast<int>(TonemapOperator::none)) {
        ImGui::TextDisabled("Clamped, as before M8");
    }

    ImGui::SliderFloat("Exposure", &render_settings_.exposure_stops, k_min_exposure_stops,
                       k_max_exposure_stops, "%+.2f stops");

    ImGui::SeparatorText("Anti-aliasing");
    ImGui::Checkbox("FXAA", &render_settings_.fxaa_enabled);
    ImGui::BeginDisabled(!render_settings_.fxaa_enabled);
    // Lower catches more edges and softens more of the image with them; higher
    // leaves faint edges alone. The floor matters most in dark regions, where a
    // tiny absolute difference is a large relative one.
    ImGui::SliderFloat("Edge threshold", &render_settings_.fxaa_edge_threshold, 0.03F, 0.33F,
                       "%.3f");
    ImGui::SliderFloat("Dark floor", &render_settings_.fxaa_edge_threshold_min, 0.005F, 0.1F,
                       "%.4f");
    ImGui::SliderFloat("Subpixel", &render_settings_.fxaa_subpixel_quality, 0.0F, 1.0F, "%.2f");
    ImGui::EndDisabled();
}

void Application::draw_lighting_menu() {
    // Lights live in the scene graph now, so this panel holds only what is
    // global. Editing one light happens in Properties, with that light
    // selected -- the same place every other node is edited.
    std::uint32_t directional = 0;
    std::uint32_t point = 0;
    for (const gpu::SceneNode& node : scene_graph_.nodes()) {
        if (!node.alive || !node.has_light) {
            continue;
        }
        (node.light.type == gpu::LightType::directional ? directional : point) += 1;
    }
    ImGui::SeparatorText("Scene lights");
    ImGui::Text("%u directional, %u point", directional, point);
    if (directional + point > gpu::k_max_lights) {
        ImGui::TextDisabled("Over the %u the shader reads; the rest are ignored.",
                            gpu::k_max_lights);
    }
    if (directional == 0) {
        ImGui::TextDisabled("No directional light: nothing casts a shadow.");
    }
    ImGui::TextDisabled("Right-click the viewport to add one.");

    ImGui::SeparatorText("Ambient");
    ImGui::DragFloat("Intensity##ambient", &render_settings_.ambient_intensity, 0.002F, 0.0F, 1.0F,
                     "%.3f");
    ImGui::TextDisabled("Flat term; no IBL yet");

    ImGui::SeparatorText("Shadow");
    ImGui::Checkbox("Enabled##shadow", &render_settings_.shadows_enabled);
    ImGui::BeginDisabled(!render_settings_.shadows_enabled);
    // Acne and peter-panning are the two ends of one trade-off: too little bias
    // and the surface shadows itself in stripes, too much and the contact
    // shadow detaches from the object. The middle is found by dragging.
    // The constant term's range dwarfs the slope term's because the spec scales
    // it by the depth format's smallest resolvable difference -- about 2^-23
    // here. Hundreds is the working range, not single digits.
    ImGui::DragFloat("Depth bias", &render_settings_.shadow_depth_bias, 10.0F, 0.0F, 10000.0F,
                     "%.0f");
    ImGui::DragFloat("Slope bias", &render_settings_.shadow_slope_bias, 0.05F, 0.0F, 8.0F, "%.2f");
    ImGui::DragFloat("Normal bias", &render_settings_.shadow_normal_bias_texels, 0.05F, 0.0F, 8.0F,
                     "%.2f texels");
    ImGui::SliderInt("PCF radius", &render_settings_.shadow_pcf_radius, 0, 4);
    ImGui::EndDisabled();
    const int taps = ((2 * render_settings_.shadow_pcf_radius) + 1) *
                     ((2 * render_settings_.shadow_pcf_radius) + 1);
    ImGui::TextDisabled("%ux%u map, %d taps", renderer_.shadow_resolution(),
                        renderer_.shadow_resolution(), taps);
}

void Application::draw_context_menu() {
    if (ImGui::BeginPopup("viewport_context")) {
        ImGui::TextDisabled("Add at %.2f, %.2f, %.2f", static_cast<double>(context_menu_point_.x),
                            static_cast<double>(context_menu_point_.y),
                            static_cast<double>(context_menu_point_.z));
        ImGui::Separator();
        if (ImGui::MenuItem("Mesh (glTF)...")) {
            // Deferred: a popup cannot be opened from inside one that is about
            // to close, so this only records the intent. draw_file_dialog,
            // which runs outside any menu, opens it.
            file_dialog_open_ = true;
            // No replace or clear from here, and the load lands at the click.
            file_dialog_scene_actions_ = false;
            file_dialog_placement_ = context_menu_point_;
        }
        if (ImGui::BeginMenu("Primitive")) {
            constexpr std::array<gpu::PrimitiveKind, 5> k_kinds{
                gpu::PrimitiveKind::plane, gpu::PrimitiveKind::cube, gpu::PrimitiveKind::sphere,
                gpu::PrimitiveKind::cone, gpu::PrimitiveKind::cylinder};
            for (const gpu::PrimitiveKind kind : k_kinds) {
                if (ImGui::MenuItem(gpu::primitive_name(kind))) {
                    // Queued, not built here: the upload blocks on a transfer
                    // submission, and this is the middle of a frame.
                    pending_primitive_ = PendingPrimitive{kind, context_menu_point_};
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Light")) {
            if (ImGui::MenuItem("Directional")) {
                pending_light_ = PendingLight{gpu::LightType::directional, context_menu_point_};
            }
            if (ImGui::MenuItem("Point")) {
                pending_light_ = PendingLight{gpu::LightType::point, context_menu_point_};
            }
            ImGui::EndMenu();
        }
        ImGui::EndPopup();
    }
}

void Application::draw_hierarchy_panel() {
    ImGui::Begin("Hierarchy");

    // Inside Begin, because GetStateStorage() returns the *current* window's,
    // and this one holds every tree node's open flag keyed by node index.
    if (hierarchy_state_stale_) {
        ImGui::GetStateStorage()->Clear();
        hierarchy_state_stale_ = false;
    }

    const std::span<const gpu::SceneNode> nodes = scene_graph_.nodes();

    if (nodes.empty()) {
        ImGui::TextDisabled("Empty scene -- load a glTF file.");
        ImGui::End();
        return;
    }

    const ChildTable table = build_child_table(scene_graph_);
    for (const std::uint32_t root : table.roots) {
        draw_hierarchy_node(root, table);
    }

    ImGui::End();
}

void Application::draw_hierarchy_node(std::uint32_t index, const ChildTable& table) {
    const gpu::SceneNode& node = scene_graph_.nodes()[index];

    // No DefaultOpen: a loaded file arrives collapsed to a single row named
    // after it, and is expanded on demand. A chess set is 50 nodes and a real
    // scene is more, which is a wall of names rather than an overview.
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    // Compared by handle rather than index: a stale handle from before a scene
    // reload must not light up whatever now occupies that slot.
    if (selected_.valid() && scene_graph_.handle_at(index) == selected_) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }
    if (table.children[index].empty()) {
        // A leaf gets no arrow, and NoTreePushOnOpen means it must not be
        // popped -- which is why the TreePop below is guarded on having
        // children rather than on `open` alone. Popping an unpushed tree node
        // corrupts ImGui's id stack and asserts somewhere unrelated later.
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    // PushID scopes the label-derived id, so two nodes sharing a name stay
    // distinct without having to build unique label strings.
    ImGui::PushID(static_cast<int>(index));
    const bool open = ImGui::TreeNodeEx(node.name.c_str(), flags);

    // Not promoted to the root, unlike a viewport click: the panel exists to
    // reach a specific node, so clicking a child selects that child and
    // outlines its own subtree.
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        select(scene_graph_.handle_at(index));
    }

    if (node.has_mesh) {
        ImGui::SameLine();
        ImGui::TextDisabled("(mesh)");
    }

    if (open && !table.children[index].empty()) {
        for (const std::uint32_t child : table.children[index]) {
            draw_hierarchy_node(child, table);
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void Application::draw_properties_panel() {
    ImGui::Begin("Properties");

    const gpu::SceneNode* node = scene_graph_.find(selected_);
    if (node == nullptr) {
        ImGui::TextDisabled("Nothing selected.");
        ImGui::TextDisabled("Click an object, or a row in the Hierarchy.");
        ImGui::End();
        return;
    }

    ImGui::Text("%s", node->name.c_str());
    ImGui::Separator();

    // Decomposed with ImGuizmo's own helper rather than glm's, so the numbers
    // shown here and the numbers a drag produces come from the same code. Two
    // decompositions that disagree on, say, euler order would make the panel
    // jitter while dragging.
    glm::mat4 local = node->local_transform;
    glm::vec3 translation{0.0F};
    glm::vec3 rotation{0.0F};
    glm::vec3 scale{1.0F};
    ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(local), glm::value_ptr(translation),
                                          glm::value_ptr(rotation), glm::value_ptr(scale));

    bool edited = false;
    bool activated = false;
    bool released = false;

    edited |= ImGui::DragFloat3("Position", glm::value_ptr(translation), 0.01F);
    activated |= ImGui::IsItemActivated();
    released |= ImGui::IsItemDeactivatedAfterEdit();
    edited |= ImGui::DragFloat3("Rotation", glm::value_ptr(rotation), 0.5F);
    activated |= ImGui::IsItemActivated();
    released |= ImGui::IsItemDeactivatedAfterEdit();
    edited |= ImGui::DragFloat3("Scale", glm::value_ptr(scale), 0.01F);
    activated |= ImGui::IsItemActivated();
    released |= ImGui::IsItemDeactivatedAfterEdit();

    // Recorded before the edit is written below, so the stored value is the one
    // the drag began from.
    if (activated && !transform_edit_.has_value()) {
        transform_edit_ = std::make_pair(selected_, node->local_transform);
    }

    if (edited) {
        // A zero on any axis makes the matrix singular, which the next
        // decomposition cannot undo -- the node would be stuck flat.
        scale = glm::max(scale, glm::vec3(1e-4F));
        ImGuizmo::RecomposeMatrixFromComponents(glm::value_ptr(translation),
                                                glm::value_ptr(rotation), glm::value_ptr(scale),
                                                glm::value_ptr(local));
        scene_graph_.set_local_transform(selected_, local);
        scene_graph_.update_transforms();
    }
    if (released) {
        commit_transform_edit();
    }

    ImGui::Separator();
    if (node->has_light) {
        ImGui::SeparatorText("Light");
        gpu::SceneLight light = node->light;

        int type = static_cast<int>(light.type);
        const bool type_changed = ImGui::Combo("Type", &type, "Directional\0Point\0");
        bool light_edited = type_changed;
        light.type = static_cast<gpu::LightType>(type);

        bool light_activated = false;
        bool light_released = false;
        light_edited |= ImGui::ColorEdit3("Colour", glm::value_ptr(light.color));
        light_activated |= ImGui::IsItemActivated();
        light_released |= ImGui::IsItemDeactivatedAfterEdit();
        light_edited |= ImGui::DragFloat("Intensity", &light.intensity, 0.1F, 0.0F, 500.0F);
        light_activated |= ImGui::IsItemActivated();
        light_released |= ImGui::IsItemDeactivatedAfterEdit();
        if (light.type == gpu::LightType::point) {
            light_edited |= ImGui::DragFloat("Range", &light.range, 0.05F, 0.01F, 500.0F);
            light_activated |= ImGui::IsItemActivated();
            light_released |= ImGui::IsItemDeactivatedAfterEdit();
        } else {
            // A directional light has no position, only a bearing, and the
            // rotate gizmo is how that is set.
            ImGui::TextDisabled("Rotate to aim; position is only the icon's.");
        }
        // The type combo commits in one go, so it is its own command rather
        // than the start of a drag.
        if (type_changed) {
            push_light_command("Change light type", selected_, node->light, light);
        }
        if (light_activated && !light_edit_.has_value()) {
            light_edit_ = std::make_pair(selected_, node->light);
        }
        if (light_edited) {
            scene_graph_.set_light(selected_, light);
        }
        if (light_released) {
            commit_light_edit();
        }
        ImGui::Separator();
    }

    ImGui::Text("Mesh: %s", node->has_mesh ? "yes" : "no");
    if (node->has_mesh) {
        ImGui::Text("Indices: %u", node->mesh.index_count);
        ImGui::Text("Material: %u", node->material_index);
    }

    ImGui::Separator();
    // Radio buttons rather than a combo: three options that change with one
    // click, and the keyboard shortcuts below mirror them.
    int operation = gizmo_operation_;
    ImGui::TextUnformatted("Gizmo");
    ImGui::RadioButton("Move (W)", &operation, static_cast<int>(ImGuizmo::TRANSLATE));
    ImGui::SameLine();
    ImGui::RadioButton("Rotate (E)", &operation, static_cast<int>(ImGuizmo::ROTATE));
    ImGui::SameLine();
    ImGui::RadioButton("Scale (R)", &operation, static_cast<int>(ImGuizmo::SCALE));
    gizmo_operation_ = operation;

    // Scale is always along the object's own axes; offering a world-space
    // scale would just be a lie about what the gizmo does.
    if (gizmo_operation_ != static_cast<int>(ImGuizmo::SCALE)) {
        ImGui::Checkbox("Local space", &gizmo_local_space_);
    }

    ImGui::End();
}

bool Application::draw_gizmo() {
    const gpu::SceneNode* node = scene_graph_.find(selected_);
    if (node == nullptr) {
        return false;
    }

    ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
    ImGuizmo::SetOrthographic(false);

    // ImGui's logical coordinates, not framebuffer pixels: the gizmo is drawn
    // through ImGui's draw list, which works in the same space the mouse does.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const VkExtent2D extent = swapchain_.extent();
    const float to_logical_x = viewport->Size.x / static_cast<float>(extent.width);
    const float to_logical_y = viewport->Size.y / static_cast<float>(extent.height);
    ImGuizmo::SetRect(viewport->Pos.x + static_cast<float>(viewport_rect_.offset.x) * to_logical_x,
                      viewport->Pos.y + static_cast<float>(viewport_rect_.offset.y) * to_logical_y,
                      static_cast<float>(viewport_rect_.extent.width) * to_logical_x,
                      static_cast<float>(viewport_rect_.extent.height) * to_logical_y);

    const CameraMatrices matrices = camera_matrices();

    // The gizmo manipulates a world transform, which is what makes dragging a
    // child behave the way the screen suggests rather than in its parent's
    // rotated frame.
    glm::mat4 world = node->world_transform;

    const bool changed = ImGuizmo::Manipulate(
        glm::value_ptr(matrices.view), glm::value_ptr(matrices.projection_gl),
        static_cast<ImGuizmo::OPERATION>(gizmo_operation_),
        gizmo_local_space_ ? ImGuizmo::LOCAL : ImGuizmo::WORLD, glm::value_ptr(world));

    // A drag runs over many frames and writes a transform on each. Recording
    // where it started and pushing once on release is what makes Ctrl+Z undo
    // the drag rather than one frame of it.
    if (ImGuizmo::IsUsing() && !transform_edit_.has_value()) {
        transform_edit_ = std::make_pair(selected_, node->local_transform);
    }
    if (!ImGuizmo::IsUsing()) {
        commit_transform_edit();
    }

    if (changed) {
        // Back out of world space into the parent's. The parent's world
        // transform is already final this frame -- parents precede children --
        // so no re-composition is needed before inverting it.
        glm::mat4 parent_world{1.0F};
        if (const gpu::SceneNode* parent = scene_graph_.find(node->parent); parent != nullptr) {
            parent_world = parent->world_transform;
        }
        scene_graph_.set_local_transform(selected_, glm::inverse(parent_world) * world);
        scene_graph_.update_transforms();
    }

    return ImGuizmo::IsUsing();
}

}  // namespace sage::app
