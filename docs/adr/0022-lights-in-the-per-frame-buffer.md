# 0022: Lights in the per-frame buffer, BRDF as free functions

**Status:** Accepted

**Context:** M5 replaces a light direction hardcoded in the fragment shader with
a real light list, and half-Lambert with Cook-Torrance GGX. Two shapes had to be
chosen. Lights could live in their own device-local storage buffer, like the
material table, or in the per-frame buffer alongside the camera. And the BRDF
could be written inline in `fragment_main`, which is the shorter route, or as
standalone functions.

**Decision:** Lights go in `FrameData` as a fixed array of eight with a count,
not a storage buffer of their own. They change every frame, which is what the
per-frame buffer already exists for, and reusing it means no new buffer, no new
bindless slot and no second slotted-write path. The material table went
device-local for the opposite reason — it is read per fragment and written once,
so host-visible reads would cross PCIe; a light list is small enough that the
same argument does not bite. The cost is a fixed ceiling of eight lights, which
is honest for M5 and becomes a storage buffer the day a scene needs more.

`Light` pairs every `float3` with the scalar that follows it — `position`/`range`,
`direction`/`intensity`, `color`/`type`. That is not cosmetic. Verified against
compiled SPIR-V, a struct of bare `float3`s strides at 48 under scalar layout but
64 under std430, with different member offsets in each; pairing each vector with
a scalar makes both layouts produce `0/12/16/28/32/44` at stride 48. The struct
can therefore move between a device address and a descriptor without a single
offset changing, which is what a deferred lighting pass or a ray-tracing hit
shader would need.

The BRDF is a set of free functions — `distribution_ggx`, `geometry_smith`,
`fresnel_schlick`, `evaluate_brdf` — taking normal, view, light direction and
material parameters, returning radiance excluding the light's own contribution.
Inlining would have been shorter and is the reason to write it out: a deferred
lighting pass has no draw context and a hit shader has no rasterizer, but both
can call this unchanged.

**Consequences:** Eight lights is a hard limit baked into `FrameData`'s layout,
and `k_max_lights` is duplicated in C++ and Slang with nothing checking they
agree — the hazard [ADR 0007](0007-one-bindless-descriptor-set.md) already
records for the bindless binding numbers. `geometry_smith` uses the direct-lighting
roughness remap `k = (r+1)^2/8`; IBL prefiltering needs `k = r^2/2`, so M7 must
not reuse it as-is. Point lights use windowed inverse-square falloff rather than
plain `1/d^2`, which never reaches zero and would leave every light influencing
every fragment forever. Specular now routinely exceeds 1.0 and clamps against the
sRGB swapchain; that is knowingly deferred to M6's HDR target, and it is the
reason a bright point light blows out to flat white rather than tonemapping.
