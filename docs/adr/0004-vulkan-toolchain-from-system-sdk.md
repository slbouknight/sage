# 0004: Vulkan toolchain from the system LunarG SDK, not vcpkg

**Status:** Accepted

**Context:** M1 needs Vulkan headers, the loader, and validation layers; vcpkg
offers `vulkan-headers`/`vulkan-loader`/`vulkan-validationlayers` ports, but
reassembling them there risks the classic Linux footgun of a loader and layer
manifests that disagree. The LunarG SDK already ships all of it — plus
`slangc`, which M2 needs — as one versioned, coherent release.

**Decision:** `find_package(Vulkan REQUIRED)` resolves against `$VULKAN_SDK`
(set by the SDK's `setup-env.sh`); CMake hard-fails with an actionable message
if that variable is unset. GLFW and VMA still come from vcpkg — neither port
depends on `vulkan-headers`, so they bind to whatever `find_package(Vulkan)`
found and no header-version divergence is possible.

**Consequences:** Contributors must source `setup-env.sh` before configuring,
and CI must install the same pinned SDK version. In exchange, the loader,
layers, and shader compiler always match each other.
