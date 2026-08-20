# 0017: One shared sampler, anisotropy on

**Status:** Accepted

**Context:** glTF lets every texture name a sampler with its own filter and wrap
modes. Honouring that means a `VkSampler` per distinct glTF sampler, a lookup
from texture to sampler, and descriptors in the bindless array that are no
longer interchangeable. Lantern declares no `samplers` array at all, so every
texture in the only asset we have falls back to the spec defaults.

**Decision:** One `Sampler` shared by every texture: `LINEAR` min and mag,
`VK_SAMPLER_MIPMAP_MODE_LINEAR`, `REPEAT` on all three axes, and
`maxLod = VK_LOD_CLAMP_NONE` so one sampler serves textures with different mip
counts. `samplerAnisotropy` is enabled at device creation with
`maxAnisotropy` clamped to `VkPhysicalDeviceLimits::maxSamplerAnisotropy` (16 on
this hardware). Anisotropy is requested now rather than later specifically
because it is a `VkPhysicalDeviceFeatures` bit — it can only be asked for when
the logical device is created, so deferring means reopening `device.cpp`. As
with every other feature here, the enable in `Device` is paired with a `require`
in `select_physical_device`: enabling an unadvertised feature is undefined
behaviour, not a reported error.

**Consequences:** glTF `TEXTURE_WRAP_CLAMP_TO_EDGE` and nearest-neighbour
filtering are silently ignored — an asset relying on either renders wrong, which
is acceptable while no asset does. Trilinear plus anisotropy is what makes the
mip chain visible as an improvement rather than as blur: without
`MIPMAP_MODE_LINEAR` a seam slides across surfaces as the camera dollies, and
without anisotropy every grazing-angle surface blurs along the axis that did not
need it. Per-sampler support later is additive — a small cache keyed on the
glTF sampler index — and does not disturb the bindless array's shape.
