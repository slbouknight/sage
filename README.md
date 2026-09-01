# sage

Standalone Vulkan renderer, built as a from-scratch exploration of modern GPU
engineering: dynamic rendering (VK 1.3), bindless resources,
buffer-device-address-driven data, and a physically based forward shading path.
Linux-only, C++20, Clang-first.

## Status

**M5 — BRDF and lights.** Loads a textured glTF scene and renders it with a
Cook-Torrance BRDF under a data-driven light list, with a fly camera.

Built so far:

- **Device** — instance with validation, physical-device selection against an
  explicit required-feature set, graphics/present/transfer queues, VMA,
  swapchain, timeline-semaphore frame pacing.
- **Rendering** — dynamic rendering only (no `VkRenderPass` or
  `VkFramebuffer`), Slang compiled to SPIR-V at build time, a pipeline cache
  persisted across runs, depth buffering.
- **Resources** — one bindless descriptor set (`UPDATE_AFTER_BIND`), vertex
  data reached by buffer device address, per-draw state in push constants.
- **Geometry** — a single device-local suballocated vertex/index buffer fed by
  transfer-queue staging uploads with queue-family ownership transfer; glTF
  parsed via fastgltf, node hierarchy flattened at load time.
- **Materials** — glTF base-colour maps decoded with stb, uploaded with layout
  transitions and a blit-generated mip chain, and registered into the bindless
  sampled-image array; material factors in a device-local storage buffer,
  selected per draw by an index in push constants.
- **Shading** — Cook-Torrance GGX on the metallic-roughness workflow, with
  tangent-space normal mapping and emissive; sRGB decoded on texture read and
  encoded on attachment write, so lighting maths runs in linear space, while
  normal and metallic-roughness maps stay linear.
- **Lights** — directional and point lights with windowed inverse-square
  falloff, read from a per-frame buffer addressed by device address rather than
  baked into the shader.

No shadows, IBL, HDR target or editor yet — v1.0 is an editor shell with
selection, gizmos and runtime glTF loading. See [`CLAUDE.md`](CLAUDE.md) for
the milestone ladder and hard project constraints, and [`docs/adr/`](docs/adr/)
for the reasoning behind non-obvious decisions — including
[ADR 0024](docs/adr/0024-vulkan-only-cuda-cut.md), which records why CUDA interop
was scoped, seriously considered, and then cut.

## Running

```bash
scripts/build.sh debug --run
```

That opens on an empty scene. Test models are fetched rather than committed:

```bash
python3 tools/fetch_assets.py
```

Then load one from the **Load glTF** panel, or name it on the command line:

```bash
./build/debug/src/app/sage assets/lantern/Lantern.gltf
```

See [`assets/README.md`](assets/README.md) for what the three models are and
why they are not in the repository.

The camera frames itself on whatever it loads. Controls follow Unreal's
viewport: **hold right mouse** to look, **WASD** to fly, **E**/**Q** for
up/down, **scroll** to change speed.

## Building

Requires Clang, GCC, Ninja, CMake ≥ 3.28,
[vcpkg](https://github.com/microsoft/vcpkg) (`VCPKG_ROOT` set in your
environment), and the [LunarG Vulkan SDK](https://vulkan.lunarg.com/).

The SDK must be on the environment before configuring — `find_package(Vulkan)`
resolves against `VULKAN_SDK`, and the validation layers are found through the
layer path the same script exports (see
[ADR 0004](docs/adr/0004-vulkan-toolchain-from-system-sdk.md)):

```bash
source ~/vulkansdk/<version>/setup-env.sh
```

Add that to your shell profile to avoid repeating it. Configure fails with an
explicit message if `VULKAN_SDK` is unset.

[ccache](https://ccache.dev/) is used automatically for every preset when it is
installed, and silently skipped when it is not — it is a speedup, not a
requirement.

Shaders are compiled by `slangc`, which local builds take from the Vulkan SDK.
CI installs a pinned [standalone Slang release](https://github.com/shader-slang/slang/releases)
instead, since the SDK is not packaged for apt and CI needs only the compiler.
Either is found via `PATH`.

```bash
cmake --preset debug && cmake --build --preset debug
ctest --test-dir build/debug --output-on-failure
./build/debug/src/app/sage
```

### Helper scripts

[`scripts/build.sh`](scripts/build.sh) wraps the above for any preset, and
sources the Vulkan SDK automatically if `VULKAN_SDK` is not already set:

```bash
scripts/build.sh                 # clang Debug
scripts/build.sh gcc-debug       # gcc Debug
scripts/build.sh asan --test     # sanitizers, then ctest
scripts/build.sh debug --run     # build, then launch the app
```

[`scripts/check.sh`](scripts/check.sh) runs everything CI enforces —
formatting plus both compilers built and tested — and is the thing to run
before committing:

```bash
scripts/check.sh                 # format check + clang + gcc
scripts/check.sh --fix           # reformat in place instead of failing
```

GCC is kept green as a second compiler:

```bash
cmake --preset gcc-debug && cmake --build --preset gcc-debug
```

Profile in `relwithdebinfo`, debug in `debug` — never benchmark a Debug build.
The `asan` preset adds AddressSanitizer and UndefinedBehaviorSanitizer.

## Design decisions

Non-obvious architectural calls are recorded as short ADRs in
[`docs/adr/`](docs/adr/).
