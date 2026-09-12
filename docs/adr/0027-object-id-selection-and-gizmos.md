# 0027: Selection through an object-id attachment, and ImGuizmo over it

**Status:** Accepted

**Context:** M7 needs three things that look separate and are not: knowing what
the cursor is over, drawing an outline around it, and dragging it. The milestone
called for an `R32_UINT` object-id attachment serving the first two, which also
rules out the usual stencil-based outline — the depth format is `D32_SFLOAT` and
carries no stencil.

**Decision:** The scene pass writes a second colour attachment, `IdBuffer`, of
`R32_UINT` at swapchain size, holding node index + 1 per pixel with 0 reserved
for empty space. Picking copies one texel to a host-visible buffer and blocks on
the submission; the alternative, ring-buffering across frames in flight, would
hand back what was under the cursor two frames ago, and the stall costs only the
frame a click happens on. The index resolves through `SceneGraph::handle_at` to
a generation-checked handle rather than being kept raw, so a pick that survives
a scene reload compares unequal instead of naming whatever now occupies the slot.

A viewport click selects `root_of` whatever it hit. That is what makes a loaded
file behave as one object to drag, and it is also what the properties panel
wants to describe — a submesh's transform is rarely the thing being edited. The
hierarchy panel deliberately does *not* promote: it exists to reach a specific
node.

Outlining a subtree is what forced the one non-obvious piece. A single selected
id in a push constant was enough while a click selected one mesh, but
`SceneGraph` guarantees only that parents precede children, **not** that
descendants are contiguous, so a range test would be wrong the moment a child is
added to an older parent — which `add_node` permits. Membership is therefore
precomputed per node by `mark_subtree`, one forward pass exploiting the same
invariant that makes `update_transforms` a flat loop, and uploaded to
`SelectionBuffer` at bindless storage slot 1. The outline shader does a lookup
where it used to do a comparison.

The outline test is "not selected, but a neighbour is", which puts the line just
outside the silhouette; the inverse test draws over the model's own edge pixels
and reads as if the mesh had been trimmed. Reading ids rather than depth or
normals means the edge is correct by construction — no threshold, no false edges
on creases, and self-intersecting geometry outlines as one shape.

**Consequences:** The id attachment needs its own bindless binding. A shader's
sampled type must match the format's numeric type, so a `uint` image cannot
share the `float` colour-texture array. It is `SAMPLED_IMAGE` rather than
`COMBINED_IMAGE_SAMPLER` because `Load()` takes integer coordinates and does no
filtering, which suits an id — interpolating two of them would produce a third,
meaningless one — and sidesteps integer formats supporting no filtering at all.

Pass order is load-bearing: scene, then outline, then pick copy. That gives each
barrier exactly one source layout (`COLOR_ATTACHMENT` → `SHADER_READ_ONLY` →
`TRANSFER_SRC`) instead of making either pass handle both cases, so the outline
pass runs its barrier even with nothing selected.

Two traps worth recording. Giving the id attachment a narrower `colorWriteMask`
than the colour attachment requires the `independentBlend` device feature —
without it every attachment's blend state must match element for element. Both
entries are identical instead: mask bits for components a format does not have
are ignored, so matching costs nothing. And ImGuizmo assumes GL-style NDC while
`perspective_vk` flips Y for Vulkan's; handing it the rendering projection puts
the gizmo upside down. `perspective_gl` exists for that, and the frame capture
that confirmed it showed the green axis extending upward from the object's
projected origin, which is exactly the thing the flip would have inverted.

The gizmo manipulates a world transform and writes back
`inverse(parent_world) * world`, so dragging a child behaves the way the screen
suggests rather than in its parent's rotated frame. Dragging suppresses picking,
or releasing over another object would reselect it.

None of the GPU-side work is unit-testable — it all needs a live device — so it
was verified by capturing the framebuffer and counting pixels: 4540 outline
pixels spanning the whole helmet for a root selection against 764 confined to
the lenses for a child, and the gizmo's three axis colours converging on the
object's projected origin. `SceneGraph`'s share *is* testable and is covered,
including the interleaved case where a subtree is not contiguous.
