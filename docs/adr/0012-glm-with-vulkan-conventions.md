# 0012: glm for math, with Vulkan clip-space conventions forced

**Status:** Accepted

**Context:** M3 needs view and projection matrices. glm is the de-facto standard
for this work, but it targets OpenGL's conventions: clip-space z in [-1, 1] and
+Y up. Vulkan uses z in [0, 1] and a framebuffer Y axis pointing down. Getting
the z range wrong makes everything fail or pass the depth test; getting Y wrong
renders the scene mirrored, which is easy to miss on symmetric geometry.

**Decision:** Add glm, and correct both conventions in exactly one place each.
`GLM_FORCE_DEPTH_ZERO_TO_ONE` is a `PUBLIC` compile definition on `sage_core`
rather than a `#define` in a header, because it must be identical in every
translation unit that includes glm — differing definitions are an ODR violation
that silently yields different matrices per TU, and a header-level define only
works if that header is included first. The Y flip lives in
`core::perspective_vk`, which negates `projection[1][1]`; the alternative, a
negative-height viewport, would split the fix across two files.

**Consequences:** `<glm/...>` is wrapped by `sage/core/math.hpp` and should not
be included directly, matching the rule already applied to spdlog and VMA. The
Y negation reverses apparent winding, so `frontFace` must be
`COUNTER_CLOCKWISE` for geometry wound counter-clockwise when viewed from
outside — the two settings are coupled, and changing one alone silently inverts
culling. Note also that glm's `mat4` has alignment 4, not 16, so any struct
shared with a shader needs `alignas(16)` on matrix members plus `offsetof`
assertions; `sizeof` alone does not catch the mismatch.
