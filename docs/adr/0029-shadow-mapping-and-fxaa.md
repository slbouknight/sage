# 0029: Shadow mapping, FXAA, and the end of v1.0

**Status:** Accepted

**Context:** Unshadowed PBR reads as flat however correct the BRDF is, which is
why M9 was always the last milestone. The scope was cut from the ladder's
cascaded shadow maps to a single directional map: the v1.0 hero scene is a
chess set, where one frustum fitted to the scene is not an approximation of
cascades but identical to them. FXAA was added alongside, because a still with
stepped edges reads as amateur no matter what is in it, and M8's HDR pipeline
had left the machinery a post-process needs already in place.

**Decision:** One `ShadowMap`, `D32_SFLOAT` at 2048², read through a
*comparison* sampler -- `compareEnable` with `LINEAR` filtering, so the hardware
filters the results of four depth tests rather than four depths. Averaging raw
depths would compare against a surface that exists nowhere in the scene. Every
PCF tap is therefore already a 2×2, and radius 2 is a 5×5 grid of filtered
tests. The border is `CLAMP_TO_BORDER` with opaque white so a fragment outside
the light's frustum reads as lit; `CLAMP_TO_EDGE` would smear the edge texel's
occlusion outwards and streak shadows across everything beyond the map.

The frustum is refitted every frame from live world transforms, which is what
`MeshView`'s new local AABB is for. A world AABB remembered at load time goes
stale the moment the gizmo moves anything, and the shadow would stop following
its caster. It fits the bounding *sphere* rather than the box, so the frustum
keeps one size as the light direction swings and the shadow's resolution does
not change while a slider is dragged.

FXAA needs an image it is not also writing, hence `LdrTarget` between the
tonemap and the swapchain. Its format is `R8G8B8A8_UNORM` where everything
around it is `_SRGB`, and that is the load-bearing decision: FXAA finds edges by
comparing luma, and luma only matches what the eye calls an edge when it is
perceptual. An `_SRGB` image would encode on write and decode on read, handing
the filter linear values and defeating the point twice. So the tonemap applies
the sRGB transfer function itself, the UNORM store keeps it verbatim, and the
FXAA pass undoes it on the way out because the swapchain's colour write will
encode again.

FXAA rather than MSAA, and not only for cost. This renderer writes an
`R32_UINT` object-id attachment alongside colour, and resolving that is
meaningless: averaging two object ids yields a third belonging to neither,
breaking picking and outlining together. MSAA would mean multisampling every
attachment and inventing a resolve rule for the id one. FXAA is a pass that
touches nothing else.

**Consequences:** Two default values were wrong in ways that looked fine and
were only caught by measuring.

The constant depth-bias factor was set to 1.5, which on this hardware is
*indistinguishable from zero* -- 28145 shadowed pixels against 28129 with bias
off entirely. The spec's offset is `m · slopeFactor + r · constantFactor`, and
the two terms are in different units: `m` is the depth slope, but `r` is the
smallest resolvable depth difference, around 2⁻²³ for a float depth map. A
constant factor of 1.5 is worth about 1e-7. Visible change begins in the
hundreds; the default is now 500. The slope term is not scaled by `r`, which is
why it worked when the constant did not.

The normal-offset bias was in world units at 0.02, which is four percent of the
chess set's radius and wiped out 88% of the shadowed pixels (5.46% of the frame
down to 0.64%). It is now measured in shadow-map texels and converted against
the fitted frustum, so one default holds at any scene scale. An absolute world
distance cannot: the same number is nothing on a cathedral and everything on a
chess piece.

A third default was wrong for a different reason. M5's key-light intensity of
2.0 left a typical model around 45/255, where a shadow has no room to be darker
than its surroundings -- the shadows were correct and invisible. Raising it to
25 put the subject at 124 with shadows a clear 98 levels below. The default is
now 8, and the light is angled rather than near-vertical, which had been putting
every shadow directly underneath the object casting it.

None of this is unit-testable, so it was verified by rendering a difference mask
between shadowed and unshadowed frames and confirming the dark regions were cast
shadows in the direction the sun implied, and by a 5× crop across a silhouette
for FXAA. One measurement did survive as a check rather than a discovery: with
FXAA disabled the new pipeline reproduces a capture taken before FXAA existed to
within a maximum channel delta of 1, with no channel differing by more than
that, which is 8-bit rounding and confirms the sRGB encode and decode cancel
exactly.

The editor gained what capturing all of this needed: F11 to hide the panels and
give the 3D view the whole window, and a capture mode that either excludes the
editor -- the portfolio frame -- or includes it, which is the only way to show
that the thing is an editor at all.
