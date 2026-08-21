# 0019: view_projection in a per-frame buffer, model in push constants

**Status:** Accepted

**Context:** Since M2 the CPU premultiplied `projection * view * model` and
pushed the result as one matrix. That is sufficient to place a vertex in clip
space and nothing else: the product is not factorable, so the shader cannot
recover `model` alone, and `model` alone is what puts a position in world space
and rotates a normal into it. The vertex shader therefore passed normals through
untransformed and lit them in object space. This was already wrong — Lantern's
root node carries `rotation: [0, 1, 0, 0]`, a 180° rotation about Y inherited by
all three meshes, so every normal in the scene was reflected in X and Z. It was
invisible only because all three meshes shared the *same* error, which reads as
"the light is somewhere else" rather than as a bug. Keeping both matrices in
push constants was not an option: 64 + 64 + an 8-byte vertex pointer is 136
bytes against the 128 that `maxPushConstantsSize` is guaranteed to provide.

**Decision:** Split the transform chain by rate of change. `view_projection` and
`camera_position` move into a `FrameData` block written once per frame and read
by every draw; `model` stays in push constants, which is the per-draw channel.
The block is reached by device address per [ADR 0008](0008-buffer-device-addresses-for-buffer-data.md),
so a draw carries an 8-byte pointer rather than a 64-byte copy. The buffer holds
`FramePacer::k_frames_in_flight` slots and each frame writes its own, indexed by
`FrameContext::slot`: the camera moves every frame, and one shared slot would be
overwritten while the GPU still reads the previous frame's copy. `begin_frame()`
already waits out the work that used a slot, which is exactly the guarantee that
makes N slots sufficient. `Buffer::write` and `Allocator::flush` gained offset
parameters to address a single slot; both default to zero, so existing callers
are untouched. This is also the partitioning classic Vulkan expresses through
descriptor set frequency — per-frame set 0, per-object set 2 — reached by a
different mechanism rather than a different idea.

**Consequences:** Push constants now use 80 of 128 bytes, leaving room for the
material index that materials will add. The frame address is *not* constant: it
alternates with the slot, so it must be recomputed inside the frame rather than
hoisted to construction — hoisting it reintroduces exactly the race the slots
exist to prevent. `FrameData` needs `alignas(16)` on both members because glm's
`mat4` has alignment 4 (see [ADR 0012](0012-glm-with-vulkan-conventions.md));
note also that its C++ `sizeof` is 80 while the SPIR-V `ArrayStride` is 76 under
scalar layout, which is harmless only because a single element is ever
dereferenced — a `FrameData*` must never be indexed as an array. The vertex
shader now does two `mat4 x vec4` products instead of one, and the CPU stops
doing a `mat4 x mat4` per draw per frame. Normals are transformed by
`mat3(model)`, correct under rotation and uniform scale but shearing under
non-uniform scale; the inverse transpose does not fit the remaining push-constant
budget and waits for a per-object buffer. `Buffer` had no users after M3 replaced
its only one; this revives it rather than adding a new type.
