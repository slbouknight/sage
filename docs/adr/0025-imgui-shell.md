# 0025: ImGui shell, drawn in its own dynamic-rendering scope

**Status:** Accepted

**Context:** M6 needs a UI, both for the editor v1.0 is aimed at and because
tuning lights and materials has meant editing constants and rebuilding since M5 —
the friction [ADR 0020](0020-brdf-and-ibl-milestones.md) predicted. ImGui is the
obvious choice, but it arrives with a hard constraint: this project forbids
`VkRenderPass` and `VkFramebuffer` outright, and ImGui's Vulkan backend was
historically render-pass shaped.

**Decision:** Add `imgui` with `docking-experimental`, `glfw-binding` and
`vulkan-binding`. Docking is taken now rather than later because v1.0 is an
editor with panels, and switching the feature afterwards changes which upstream
branch vcpkg builds. Verified before committing to it: `ImGui_ImplVulkan_InitInfo`
exposes `UseDynamicRendering`, and reading the backend's implementation confirmed
it never calls `vkCmdBeginRendering` for the main viewport — it only builds a
pipeline from a supplied `VkPipelineRenderingCreateInfo` and records draw
commands into whatever scope the caller opened. So the prohibition survives
intact, and the `VK_KHR_dynamic_rendering` device extension its header comment
demands is only needed for the multi-viewport path, which is not enabled.

`ImGuiLayer` opens its own rendering scope with `loadOp = LOAD` and no depth
attachment, after the scene and before the transition to `PRESENT_SRC_KHR`. That
transition moved out of `record_scene` into `transition_to_present` precisely so
the UI can get into the swapchain image first. The backend creates its own
descriptor pool via `DescriptorPoolSize`, which is one fewer thing to size and
destroy here.

Unlike spdlog and VMA, **`<imgui.h>` is deliberately not wrapped**. Only the two
backend headers are confined to `imgui_layer.cpp`. The wrapping rule exists to
keep a swappable dependency behind a seam; ImGui's widget API *is* the thing
being used, and forwarding several hundred functions to preserve a rule would
cost more than it protects.

**Consequences:** `ImGuiLayer` must be constructed after `Window`, because
`ImGui_ImplGlfw_InitForVulkan` with `install_callbacks = true` chains to whatever
GLFW callbacks are already registered — build it first and scroll-to-adjust-speed
silently stops working. It must also be the *last* declared member of
`Application`, so it is destroyed first, while the device it frees objects
through is still alive. `color_format_` is a member rather than a constructor
local because `VkPipelineRenderingCreateInfo::pColorAttachmentFormats` is a
pointer that ImGui retains without owning. `ImGuiLayer::begin_frame()` is called
only after the swapchain acquire succeeds: the out-of-date path bails without
submitting, and an ImGui frame left open there trips the next `NewFrame()`.
Camera input is gated on `wants_mouse()`, without which dragging a slider also
spins the view. As of ImGui 1.92 the render-pass and pipeline-rendering fields
moved from `InitInfo` onto a `PipelineInfoMain` sub-struct and an `ApiVersion`
field became required — nearly every example still published shows the old
layout, so this is worth remembering the next time the dependency moves.

**Amended in M6:** the panels are docked rather than positioned. Placing each
one by hand worked for two and broke as soon as the file picker was a third —
absolute coordinates do not compose, and M7 adds a fourth. `draw_dockspace`
lays a `DockSpaceOverViewport` with `PassthruCentralNode` (the middle node stays
transparent and, while empty, passes mouse input through, so the scene shows and
the camera still works) and `NoDockingOverCentralNode` (it stays empty, so no
panel can be dragged over the 3D view). The default arrangement is built once
through the `DockBuilder` API, which lives in `imgui_internal.h` and has no
public equivalent; that include is confined to `application.cpp` and used for
nothing else. Layout is rebuilt each run rather than persisted, because
`io.IniFilename` is deliberately null. This is what the `docking-experimental`
feature was taken for above.

Docking also forced the scene to stop rendering to the whole swapchain image.
The panels cover about 40% of the width, so a model framed against the full
image was both off-centre and cropped — it looked drastically over-zoomed, and
the cause was a projection describing a view wider than the visible one, not the
framing distance. `record_scene` now sets its viewport and scissor from the
central node's rect, converted from ImGui's logical coordinates to framebuffer
pixels (identical until display scaling, and silently wrong afterwards).
`frame_camera_on` derives its aspect from the same rect and fits against
`min(half_fov_x, half_fov_y)`, since a near-square viewport makes the horizontal
field of view the tighter one; it also fits the bounding sphere tangentially
with `radius / sin` rather than fitting only its equatorial disc with
`radius / tan`, which is what the old 1.5 fudge factor was compensating for.
Framing is queued and applied just after `draw_dockspace`, because the aspect it
needs does not exist when the constructor loads a model named on the command
line. The rect this establishes is also what M7's cursor picking will need to
convert a mouse position into a viewport coordinate.
