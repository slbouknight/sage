# 0007: One bindless descriptor set for the whole renderer

**Status:** Accepted

**Context:** Per-draw descriptor binding costs a `vkCmdBindDescriptorSets` per
object, forces draws to be sorted by material, and requires allocating and
tracking a set per object. It also keeps resource selection on the CPU, which
blocks GPU-driven rendering later.

**Decision:** A single `BindlessSet` holds unbounded arrays of storage buffers
and sampled images (1024 each, far below the ~1M the target hardware allows).
It is bound once per frame at set 0; shaders select resources by index
delivered through push constants. Bindings use `PARTIALLY_BOUND` (so unwritten
slots are legal), `UPDATE_AFTER_BIND` (so descriptors can be written while the
set is bound), and `UPDATE_UNUSED_WHILE_PENDING`. The layout and pool carry the
matching `UPDATE_AFTER_BIND_POOL` / `UPDATE_AFTER_BIND` creation flags.

**Consequences:** Binding cost stops scaling with draw count, and because
indices are plain integers they can later come from GPU-written buffers.
`VARIABLE_DESCRIPTOR_COUNT` is enabled but unused — capacities are fixed in the
layout, which is simpler and costs only the reserved pool budget (~100 KB).
The binding numbers are duplicated in shader source and C++ constants with
nothing checking they agree; a mismatch is a runtime error, not a build error.
