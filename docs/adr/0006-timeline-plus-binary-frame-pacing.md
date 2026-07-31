# 0006: Timeline semaphore pacing alongside binary WSI semaphores

**Status:** Accepted

**Context:** Vulkan 1.3 has timeline semaphores in core, which replace the
usual per-frame fence array for CPU/GPU throttling with one monotonic counter
that can also be queried without blocking. They cannot replace binary
semaphores everywhere: `vkAcquireNextImageKHR` and `vkQueuePresentKHR` are
specified against `VK_SEMAPHORE_TYPE_BINARY`, because the presentation engine
is a separate agent outside the application's own submission timeline.

**Decision:** One timeline semaphore in `FramePacer`, incremented once per
submit, gates reuse of each of the two in-flight frame slots. Binary
semaphores remain for the two WSI handoffs. `render_finished` is allocated per
**swapchain image** rather than per frame-in-flight, since the presentation
engine may still hold an older frame's semaphore when a slot is recycled and
image count need not equal frames-in-flight.

**Consequences:** Two synchronization primitives instead of one, which is more
to explain but reflects two genuinely different problems. The timeline counter
gives later milestones non-blocking frame-completion queries that fences could
not.
