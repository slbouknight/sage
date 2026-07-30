# 0001: Vulkan-only, no `rhi/` abstraction layer yet

**Status:** Accepted

**Context:** A second render backend isn't in scope until M5, and the seam a
real RHI needs is only visible once CUDA interop shows which parts of `gpu/`
actually vary. Building the abstraction now means guessing at that seam.

**Decision:** `gpu/` talks to Vulkan directly — no `IDevice`/`ICommandBuffer`
interfaces, no backend enum, and no `VkRenderPass`/`VkFramebuffer` (dynamic
rendering only, VK 1.3). Revisit at M5 when CUDA interop lands.

**Consequences:** Faster to build M1–M4; risk of Vulkan-specific assumptions
leaking into call sites that will need to move behind the eventual seam.
