# SAGE

Standalone Vulkan renderer. C++20, Linux-only, Clang-first.
Portfolio project targeting GPU/simulation engineering roles — code quality and
documented reasoning matter as much as features. CUDA was cut after M5; see
[ADR 0024](docs/adr/0024-vulkan-only-cuda-cut.md).

## Hard prohibitions

- **No `VkRenderPass` or `VkFramebuffer`, ever.** Dynamic rendering only (VK 1.3).
- **No source globbing** in CMake. List files explicitly.
- **No `rhi/` abstraction layer.** `gpu/` talks to Vulkan directly. This was once
  "not yet", pending a seam CUDA interop would reveal; with CUDA cut there is no
  second backend and no seam to find, so it is now "not at all". Do not add
  interfaces for a backend that will never exist.
- **No new dependencies** without asking. Current set is locked in `vcpkg.json`.
- **No `#include <spdlog/...>` outside `sage/core/log.hpp`.** Same for any third-party
  header we've wrapped.
- **No CUDA.** Cut after M5 — see [ADR 0024](docs/adr/0024-vulkan-only-cuda-cut.md).
  Ray tracing, if it ever lands, is `VK_KHR_ray_query`, not a second API.
- Do not add OptiX, TensorRT, or neural denoising. Deferred by decision.
- **No OpenUSD.** Cut from the roadmap in M3 — see
  [ADR 0003](docs/adr/0003-defer-openusd-behind-flag.md). It is pipeline/DCC work,
  not renderer work, and belongs in a separate project.

## Milestone ladder

- **M0** — Build skeleton: CMake/Ninja/Clang, presets, vcpkg manifest, format/tidy, CI.
  `core/`: logging, assert, handle/GUID, job stub. No graphics.
- **M1** — Device bring-up: instance, validation, device selection w/ explicit required
  features, VMA, swapchain, timeline-semaphore frame pacing. Output: cleared screen.
- **M2** — Triangle: dynamic rendering, Slang→SPIR-V at build time, pipeline cache,
  one bindless set (`UPDATE_AFTER_BIND`), push constants carrying indices + BDA pointers.
- **M3** — Geometry registry: single suballocated vertex/index buffer, transfer-queue
  staging uploads, glTF via fastgltf. Done, split in two:
  - **M3.1** — upload path proven against a hardcoded cube. Also picked up depth
    buffering, glm, and an Unreal-style fly camera — the camera was scope added
    deliberately, since verifying an arbitrary loaded model needs more than one
    viewing angle. It is *not* the editor's work: no ImGui, no gizmos, no selection.
  - **M3.2** — fastgltf, on top of an already-proven upload path.
- **M4** — Materials and textures: image upload path (layout transitions, mip
  generation), samplers, base-colour textures registered into the bindless
  sampled-image array, glTF material factors in a bindless storage buffer, and a
  per-frame camera buffer addressed by BDA so lighting works in world space.
  Inserted ahead of the editor because the sampled-image half of the bindless set
  has never been exercised, and object-space normals break the moment two objects
  rotate independently. One directional light, base colour only; the BRDF is M5's
  job and IBL is M8's.
- **M5** — BRDF and lights: Cook-Torrance GGX on the metallic-roughness workflow,
  normal mapping off `TANGENT`, the remaining three Lantern maps wired through
  the material table, and a `Light` type — directional and point, with
  attenuation — read from data rather than hardcoded as shader constants.
  The two ship together because a BRDF is evaluated per light: splitting them
  means writing the loop body with the loop removed, then re-adding it later.
  Watch the format split — base colour and emissive are sRGB, normal and
  metallic-roughness are UNORM, and getting that wrong looks almost right.
  Still no IBL: the only ambient term is a constant. Renders straight to the
  swapchain, so specular highlights above 1.0 clamp — a known and accepted
  limitation until M8 introduces the HDR offscreen target.
  Lights live in the per-frame buffer, not a storage buffer of their own; see
  [ADR 0022](docs/adr/0022-lights-in-the-per-frame-buffer.md). Done.
- **M6** — Scene graph and ImGui shell: a persistent mutable hierarchy replacing
  M3's flattened load-time `Scene`, a hierarchy panel over it, and loading
  arbitrary glTF files from a small `std::filesystem` file picker rather than one
  hardcoded model. Flat array of nodes with parent indices, ordered so parents
  precede children — world transforms then update in one linear pass and fall out
  as the draw list the renderer already consumes. Node references are
  `core::Handle`, written in M0 and unused until now: deleting a node bumps its
  generation so stale handles compare unequal rather than dangle. Forces the
  question [ADR 0011](docs/adr/0011-bump-suballocated-geometry-buffer.md)
  deferred — all three registries are bump-allocated with no free, so additive
  loading needs `reset()` on each, behind `wait_idle()`, when the scene is
  cleared. Bump allocation stays correct; it was only ever missing a rewind.
  Raise `k_geometry_capacity` while here: 4 MiB suits one lantern, not a scene.
  ImGui's Vulkan backend supports dynamic rendering via `UseDynamicRendering` and
  `PipelineRenderingCreateInfo`, so the no-`VkRenderPass` prohibition survives.
  New dependency: `imgui` (`docking-experimental`, `glfw-binding`,
  `vulkan-binding`). Done. Loading turned out to be additive as well as
  replacing, for one cursor's worth of extra work — see
  [ADR 0026](docs/adr/0026-runtime-glTF-loading.md).
- **M7** — Selection and manipulation: an `R32_UINT` object-ID attachment written
  alongside colour, giving cursor picking by one-texel readback *and* outline
  highlighting from ID discontinuity in a post pass — one buffer, both features,
  and no stencil, which matters because the depth format is `D32_SFLOAT`. Then a
  properties panel over the selection and ImGuizmo writing transforms back into
  the scene graph. New dependency: `imguizmo`. ← current
- **M8** — Presentation and capture: HDR offscreen target
  (`R16G16B16A16_SFLOAT`), a full-screen tonemap resolve, and screenshot-to-PNG
  reusing `stb_image_write` from the stb port. Speculars have clamped against the
  sRGB swapchain since M5; this is where that stops.
- **M9** — Shadow mapping, and **v1.0**: directional cascade, depth-only passes
  into an array texture, PCF filtering. Unshadowed PBR reads as flat no matter
  how correct the BRDF is, so this is the last thing standing between the editor
  and a screenshot worth showing. Then polish, document, and call it done.

Past v1.0, in no committed order — pick one and finish it rather than starting
several:

- **v2 — Forward and deferred paths.** A G-buffer and full-screen lighting pass,
  reusing M6's offscreen-target machinery. Needs material identity resolvable
  without draw context, which is why `Material` and `Light` were made
  layout-portable between scalar and std430.
- **v2 — IBL and global illumination.** HDR environment, equirectangular→cubemap,
  irradiance convolution, prefiltered specular, BRDF LUT. Brings cubemap images,
  which `Texture` does not do. The prefilter needs the IBL roughness remap
  `k = r^2/2`, not `geometry_smith`'s direct-lighting one. Until then a constant
  ambient stands in; a hemisphere term is ~20 lines if it needs to look better.
- **v2 — Ray tracing and hybrid.** `VK_KHR_ray_query` for shadows, AO or
  reflections over the raster path. The bindless set and BDA vertex access
  already suit it; the geometry buffer needs
  `VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR` and the
  device needs its features requested at creation, which is creation-time only.
- **v3 — Neural rendering.** Note this collides with the standing prohibition on
  OptiX, TensorRT and neural denoising; revisit that decision before starting.

Work the current milestone only. Flag scope creep instead of accommodating it.

## Conventions

- One static lib per module, namespaced alias: `sage::core`, `sage::gpu`, ...
- Public headers at `src/<module>/include/sage/<module>/`. Always
  `#include <sage/core/log.hpp>`, never relative paths across modules.
- `target_link_libraries` is `PRIVATE` by default. `PUBLIC` only when the dependency
  appears in a public header.
- Warning flags live only in `cmake/SageCompileOptions.cmake`, on the
  `sage::compile_options` INTERFACE target. Every target links it.
- Debug info: `-ggdb3 -fstandalone-debug -fno-omit-frame-pointer`. libstdc++, not libc++.
- Every non-obvious decision gets a ten-line ADR in `docs/adr/`.

## Commands

```bash
cmake --preset debug && cmake --build --preset debug
ctest --test-dir build/debug --output-on-failure
cmake --preset gcc-debug && cmake --build --preset gcc-debug   # keep both compilers green
```

Profile in `relwithdebinfo`, debug in `debug`. Never benchmark a Debug build.