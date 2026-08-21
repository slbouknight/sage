# 0020: BRDF and IBL become milestones, ordered around CUDA interop

**Status:** Accepted

**Context:** M4 was scoped as "one directional light; no BRDF or IBL yet", with
the editor next and CUDA interop after it. Adding realistic materials changes
that, and the obvious move — folding a BRDF into M4 — is wrong twice over: M4 is
nearly done, and it would bundle work that fails differently. It also raised the
question of whether BRDF and IBL are one milestone or two. They are not the same
size. Cook-Torrance GGX is fragment-shader maths plus more `Material` fields and
texture slots, on top of a material table and texture registry M4 already
builds; `metallicFactor` and `roughnessFactor` are parsed today and Lantern
carries `TANGENT` on every primitive, so normal mapping needs no new asset.
IBL is a subsystem: HDR environment loading, equirectangular→cubemap projection,
irradiance convolution, a prefiltered specular chain and a BRDF integration LUT,
which drag in cubemap images, a second sampler with explicit LOD control, and
offscreen passes. Two to three times the work, and a different kind of it.

**Decision:** Split them, and order the ladder M4 materials → **M5 BRDF** →
**M6 CUDA interop** → **M7 IBL** → M8 simulation → M9 editor. IBL sits after
interop rather than before it because prefiltering is a compute workload: run
before M6 it means hand-rolling render-to-cubemap-face with graphics pipelines,
and compute infrastructure that would have done the job better arrives one
milestone later. Run after, the prefilter passes become `ComputePass`'s second
real consumer, which is less code and gives the interop abstraction a non-trivial
user rather than a single tonemap. Only one milestone is inserted ahead of CUDA,
not two, because the project targets GPU/simulation work: a PBR pass is table
stakes for a renderer, while working Vulkan/CUDA interop on a shared timeline
semaphore is the part that is actually uncommon, and delaying it by two
milestones would trade the differentiator for the commodity.

**Consequences:** M4 stays base-colour only and finishes as scoped; M5 extends
the material table rather than reworking it, so `Material` gains three
`uint32_t` texture indices and needs its std430 layout re-verified against the
SPIR-V — a probe, not a rewrite. The full Lantern texture set is now
load-bearing rather than decorative, which is what justified vendoring all four
maps unmodified (see [ADR 0014](0014-vendor-a-cc0-test-asset.md)). The editor
moving to last has a real cost worth naming: M5 and M7 are parameter-tuning
milestones — roughness, metallic, light direction, exposure — and tuning those
by editing shader constants and rebuilding is miserable, as testing a single
light direction in M4 already showed. If that friction bites, a debug-only ImGui
panel is a far smaller thing than the M9 editor shell, which needs a persistent
mutable scene graph, and can be pulled forward on its own without dragging the
rest of the editor with it. That should be a deliberate decision when the pain
appears, not a quiet slide into building the editor early.

**Amended before M5 started:** a `Light` type — directional and point lights fed
from a storage buffer instead of shader constants — joins M5 rather than
becoming a milestone of its own. A Cook-Torrance evaluation is inherently per
light, so shipping the BRDF against a single hardcoded light means writing the
loop body with the loop removed and re-adding it afterwards; and the light
buffer is structurally the same as M4's material buffer, which is a shape
already proven. Folding it in also keeps exactly one milestone ahead of CUDA
interop, which is the ordering this ADR argued for. The cost is that M5 is now
the largest milestone on the ladder, and it sharpens the tooling problem above
rather than softening it: a dynamic light system that cannot be moved
interactively is only marginally better than a hardcoded one, since the
positions merely migrate from shader constants to C++ constants. Expect the
debug-panel decision to arrive partway through M5, not after it.
