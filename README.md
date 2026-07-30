# sage

Standalone Vulkan/CUDA renderer, built as a from-scratch exploration of modern
GPU and simulation engineering: dynamic rendering (VK 1.3), bindless
resources, buffer-device-address-driven data, and CUDA interop for compute
passes. Linux-only, C++20, Clang-first.

## Status

**M0 — build skeleton.** No graphics yet: CMake/Ninja/Clang toolchain, vcpkg
manifest, `core/` (logging, assertions, handles/GUIDs, a single-threaded job
stub), format/tidy config, and CI. See [`CLAUDE.md`](CLAUDE.md) for the full
milestone ladder and hard project constraints.

## Building

Requires Clang, GCC, Ninja, CMake ≥ 3.28, and
[vcpkg](https://github.com/microsoft/vcpkg) (`VCPKG_ROOT` set in your
environment).

```bash
cmake --preset debug && cmake --build --preset debug
ctest --test-dir build/debug --output-on-failure
```

GCC is kept green as a second compiler:

```bash
cmake --preset gcc-debug && cmake --build --preset gcc-debug
```

Profile in `relwithdebinfo`, debug in `debug` — never benchmark a Debug build.

## Design decisions

Non-obvious architectural calls are recorded as short ADRs in
[`docs/adr/`](docs/adr/).
