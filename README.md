# sage

Standalone Vulkan/CUDA renderer, built as a from-scratch exploration of modern
GPU and simulation engineering: dynamic rendering (VK 1.3), bindless
resources, buffer-device-address-driven data, and CUDA interop for compute
passes. Linux-only, C++20, Clang-first.

## Status

**M1 — device bring-up.** Vulkan instance with validation, physical-device
selection against an explicit required-feature set, logical device with
graphics/present/transfer queues, VMA allocator, swapchain, and
timeline-semaphore frame pacing. Output is a cleared window. `core/` provides
logging, assertions, handles/GUIDs, and a single-threaded job stub. See
[`CLAUDE.md`](CLAUDE.md) for the full milestone ladder and hard project
constraints.

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
