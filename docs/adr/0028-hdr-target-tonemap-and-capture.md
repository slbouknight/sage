# 0028: An HDR offscreen target, a tonemap resolve, and screenshot capture

**Status:** Accepted

**Context:** M5 shipped a Cook-Torrance BRDF that renders straight to the
swapchain. The swapchain is `B8G8R8A8_SRGB`, so every value above 1.0 was
clamped at the attachment, by the hardware, before anything could decide what
to do with it. A specular highlight under a light of intensity 2 passes 1.0
without trying, and two highlights of very different energy came out the same
flat white — no amount of correctness in the BRDF shows through that. M8 is
where the range stops collapsing at the wrong moment.

**Decision:** The scene shades into `HdrTarget`, an `R16G16B16A16_SFLOAT` image
at swapchain size, and a full-screen `tonemap` pass resolves it. Half float
rather than full: the spec's required-format table guarantees it as a colour
attachment and a sampled image, it holds far more range than the 8 bits it
replaces, and it costs half the bandwidth of `R32G32B32A32` for precision no
display can show. Confirmed on NVIDIA, RADV and llvmpipe before it was written.

The tonemap output is linear. The swapchain is still an `_SRGB` format, so the
colour write applies the transfer function in fixed function exactly as it did
before; encoding in the shader as well would apply it twice.

The curve is selectable at runtime — none, Reinhard, ACES — rather than baked
in. Two curves are only comparable on the same frame; comparing across a
rebuild compares two memories of an image. `none` is not a placeholder but the
pre-M8 behaviour kept as a control: selecting it shows exactly what the tonemap
is buying. Measured on the damaged helmet at +2 stops, it is 904 pixels blown
to pure white against ACES's 63, and Reinhard's 0 — Reinhard asymptotes, so
nothing ever quite reaches white, which is the per-channel desaturation that
makes it the wrong default. Exposure is exposed in stops and converted to a
linear multiplier at push time, because a stop is the unit it is reasoned about
in.

`HdrTarget` gets its own bindless binding (3) rather than a slot in the
colour-texture array at binding 1. Its format is float, so unlike the object-id
image it *could* share that array — but the array is `TextureRegistry`'s to
allocate out of, and a render target is not scene content the registry owns and
destroys on `reset()`. A separate binding keeps the two lifetimes from
touching. It is `SAMPLED_IMAGE` read with `Load()` for the same reason as the
id attachment: the resolve is exactly one texel per pixel, so there is nothing
for a filter to interpolate. Bloom, or a resolve at a different resolution,
would want a sampler; neither exists.

The outline keeps drawing into the swapchain, after the tonemap, not into the
HDR target. It is an editor affordance rather than a lit surface, and putting
it through the curve would quietly turn the colour asked for into a darker,
less saturated one.

Capture copies the tonemapped swapchain image, cropped to the viewport rect,
between the tonemap and the outline. That gives a rule worth stating: the file
holds the rendered image and none of the editor's overlays — no panels, no
gizmo, no selection outline. The swapchain needed
`VK_IMAGE_USAGE_TRANSFER_SRC_BIT`, which it had never been created with; that
was found in M7 while capturing frames to verify the gizmo, and the surface's
`supportedUsageFlags` is now checked rather than assumed, since only
`COLOR_ATTACHMENT` is guaranteed. The readback buffer is allocated only while a
capture is in flight — a permanent one is 33 MiB at 4K for a feature used a
handful of times a session. Encoding is `stb_image_write` from the stb port
already carrying `stb_image`, so no new dependency; the bytes come out of an
`_SRGB` swapchain already sRGB-encoded, which is exactly what a PNG holds, so
no transfer function is applied on the way.

**Consequences:** The scene pass's entry barrier changed shape. It used to
transition a swapchain image whose previous use was presentation; it now
transitions the HDR target, whose previous use was the last frame's tonemap
sampling it. The source stage is therefore `FRAGMENT_SHADER`, not
`COLOR_ATTACHMENT_OUTPUT` — a write-after-read, so the access mask stays `NONE`
and only the execution dependency matters.

The scene pass clears the whole image while its draws stay scissored to the
viewport. That is deliberate: it leaves every texel the tonemap will read
defined, so the resolve is one full-screen triangle with no special case for
the region behind the panels.

One hazard was found here by synchronisation validation and would not have been
found without it. The barrier restoring the swapchain from `TRANSFER_SRC` after
a capture granted `COLOR_ATTACHMENT_WRITE`, but the outline pass that follows
begins with `loadOp LOAD`, which *reads* the attachment before blending over
it. `COLOR_ATTACHMENT_READ` had to be granted as well. Ordinary validation says
nothing about this; `VK_LAYER_VALIDATE_SYNC=1` reports it on the first captured
frame.

Worth recording as an open question: deliberately deleting the barrier that
moves `HdrTarget` to `SHADER_READ_ONLY_OPTIMAL` — leaving the tonemap sampling
an image still in `COLOR_ATTACHMENT_OPTIMAL`, through a descriptor that
declares otherwise — produced no diagnostic at all, from core validation,
synchronisation validation or GPU-assisted validation. The layer was verified
to be live and reporting in the same session. Every image this renderer samples
is reached through an `UPDATE_AFTER_BIND` descriptor, so if layout checking is
skipped for those, it is skipped for all of them, and "validation is clean" is
worth less here than it looks. Not chased further, but not assumed benign.

Verified by capture rather than by the absence of errors: the written PNG shows
the helmet with correct channel order, and the three curves measurably differ
in clipped-pixel count on the same frame. Swapchain recreation was exercised 28
times in one run through a temporary hook, since nothing here can resize a
window, confirming the HDR target is recreated and its descriptor rewritten
rather than left dangling.
