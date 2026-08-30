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
