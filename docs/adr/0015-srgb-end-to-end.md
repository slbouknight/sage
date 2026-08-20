# 0015: sRGB on both ends, linear in the middle

**Status:** Accepted

**Context:** M4 samples a base-colour texture for the first time. PNG texels are
sRGB-encoded — perceptually spaced, not linearly — so multiplying them by a
lighting term computes on the wrong numbers. Until M4 nothing sampled anything,
and the shader wrote ad-hoc constants, so the swapchain's
`VK_FORMAT_B8G8R8A8_UNORM` was harmless: values went out untouched and the
display interpreted them as sRGB, which happened to look like what was intended.

**Decision:** Decode on read and encode on write, shade linearly in between.
Base-colour textures are created as `VK_FORMAT_R8G8B8A8_SRGB` so the sampler
converts to linear per fetch, and the swapchain now prefers
`VK_FORMAT_B8G8R8A8_SRGB` so the colour attachment write performs the inverse
conversion in fixed function. Both conversions are free — dedicated silicon on
the sampler and the ROP — which is why this is preferred over doing either by
hand in the shader. Only colour data gets an `_SRGB` format; normal, metallic
and roughness maps are measurements rather than colours and must stay UNORM.

**Consequences:** The existing untextured render got brighter, because the
`0.05` clear colour is now treated as a linear value and encoded on the way out
rather than being passed through. That is the fix working, not a regression, but
the clear colour was picked to *look* dark and now reads as mid-grey; the
equivalent is roughly `0.0036` linear. Mip generation stays correct because
`vkCmdBlitImage` filters in linear space for `_SRGB` formats. Getting exactly
one of the two halves wrong is the dangerous failure: the result is plausibly
wrong rather than obviously broken, and easy to mistake for a lighting bug.
