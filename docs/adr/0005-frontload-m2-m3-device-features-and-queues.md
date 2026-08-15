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

**Amended in M2:** `shaderDrawParameters` (Vulkan 1.1) was added when the first
real shader failed to load. Slang lowers HLSL's `SV_VertexID` to
`VertexIndex - BaseVertex`, and reading `BaseVertex` requires the SPIR-V
`DrawParameters` capability. This was not predictable from the milestone ladder
— it is an artifact of the shader compiler's semantics, discoverable only by
compiling a shader — so front-loading could not have covered it. The safeguard
worked as intended: validation named the missing feature exactly, at the first
`vkCreateShaderModule`. Note that `spirv-val` does not catch this class of
problem, since SPIR-V can be entirely valid while still demanding features the
device was not created with.

**Amended in M3:** `scalarBlockLayout` (Vulkan 1.2) was added for the same
reason. Slang's "natural" layout for BDA-accessed structs packs members C-style,
so `struct Vertex { float3 position; float3 color; }` places `color` at offset
12, straddling a 16-byte boundary. Standard block layout forbids that; scalar
layout permits it. The alternative — padding vertex structs to satisfy std430 —
would distort every mesh layout to suit a rule the feature simply removes.

Two features in two milestones, both discovered by compiling a real shader
rather than predicted from the ladder, suggests the honest limit of this ADR's
approach: front-loading covers what the milestone descriptions imply, not what
a specific shader compiler's lowering happens to require. That is a reason to
expect further amendments here, not a reason to abandon the approach — the cost
of each is one line plus a matching `require()` in `physical_device.cpp`.
