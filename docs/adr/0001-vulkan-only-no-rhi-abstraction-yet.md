# 0001: Vulkan-only, no `rhi/` abstraction layer yet

**Status:** Accepted

**Context:** A second render backend isn't in scope until CUDA interop, and the seam a
real RHI needs is only visible once CUDA interop shows which parts of `gpu/`
actually vary. Building the abstraction now means guessing at that seam.

**Decision:** `gpu/` talks to Vulkan directly — no `IDevice`/`ICommandBuffer`
interfaces, no backend enum, and no `VkRenderPass`/`VkFramebuffer` (dynamic
rendering only, VK 1.3). Revisit when CUDA interop lands (M6 after the M4
renumbering; originally written as M5).

**Consequences:** Faster to build the milestones before it; risk of Vulkan-specific assumptions
leaking into call sites that will need to move behind the eventual seam.
