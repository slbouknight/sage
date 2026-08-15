# SAGE

Standalone Vulkan/CUDA renderer. C++20, Linux-only, Clang-first.
Portfolio project targeting GPU/simulation engineering roles — code quality and
documented reasoning matter as much as features.

## Hard prohibitions

- **No `VkRenderPass` or `VkFramebuffer`, ever.** Dynamic rendering only (VK 1.3).
- **No source globbing** in CMake. List files explicitly.
- **No `rhi/` abstraction layer yet.** `gpu/` talks to Vulkan directly. The seam gets
  extracted around M5 when CUDA interop reveals where it actually is. Do not
  pre-emptively add interfaces for a second backend that will never exist.
- **No new dependencies** without asking. Current set is locked in `vcpkg.json`.
- **No `#include <spdlog/...>` outside `sage/core/log.hpp`.** Same for any third-party
  header we've wrapped.
- Do not add OptiX, TensorRT, neural denoising, or custom USD schemas. Deferred by decision.

## Milestone ladder

- **M0** — Build skeleton: CMake/Ninja/Clang, presets, vcpkg manifest, format/tidy, CI.
  `core/`: logging, assert, handle/GUID, job stub. No graphics.
- **M1** — Device bring-up: instance, validation, device selection w/ explicit required
  features, VMA, swapchain, timeline-semaphore frame pacing. Output: cleared screen.
- **M2** — Triangle: dynamic rendering, Slang→SPIR-V at build time, pipeline cache,
  one bindless set (`UPDATE_AFTER_BIND`), push constants carrying indices + BDA pointers.
- **M3** — Geometry registry: single suballocated vertex/index buffer, transfer-queue
  staging uploads, glTF via fastgltf. Split in two: ← current
  - **M3.1** — upload path proven against a hardcoded cube. Also picked up depth
    buffering, glm, and an Unreal-style fly camera — the camera was scope added
    deliberately, since verifying an arbitrary loaded model needs more than one
    viewing angle. It is *not* M4's work: no ImGui, no gizmos, no selection.
  - **M3.2** — fastgltf, on top of an already-proven upload path.
- **M4** — ImGui/ImGuizmo editor shell: docking, hierarchy panel, gizmo→transform writeback.
- **M5** — CUDA interop: `ComputePass` interface, exportable VMA pool, shared timeline
  semaphore. First target: tonemap or blur on the HDR image.
- **M6** — OpenUSD ingestion behind `SAGE_ENABLE_USD` (default OFF).

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