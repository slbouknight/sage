# 0005: Front-load M2/M3 device features and queue families into M1

**Status:** Accepted

**Context:** Vulkan fixes both the enabled feature set and the queue-family
layout at `vkCreateDevice` time. Enabling `dynamicRendering`/
`bufferDeviceAddress`/descriptor-indexing only when M2 first uses them, or
adding a transfer queue only when M3 needs it, would mean reopening and
re-testing M1's "finished" device-creation path twice.

**Decision:** M1 enables the full feature set the milestone ladder already
commits to (timeline semaphores and sync2 for M1; dynamic rendering, BDA,
`shaderInt64`, and the `UPDATE_AFTER_BIND` descriptor-indexing flags for M2)
and creates graphics, present, and transfer queues up front, preferring a
transfer-only family for M3's staging uploads. Features no milestone justifies
(`maintenance4`, `bufferDeviceAddressCaptureReplay`) stay off.

**Consequences:** Unused capability is requested for two milestones, and a GPU
lacking any of it is rejected earlier than strictly necessary. Both are cheap:
the features are inert until used, and the target class of hardware supports
them universally.
