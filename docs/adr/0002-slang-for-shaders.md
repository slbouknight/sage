# 0002: Slang for shaders, compiled to SPIR-V at build time

**Status:** Accepted

**Context:** M2 needs a shader path to SPIR-V. Raw GLSL/HLSL each lock into
one ecosystem's tooling; Slang targets SPIR-V directly, has first-class
support for the bindless/buffer-device-address patterns this project already
commits to, and gives a single source language if CUDA kernels ever want
shared math code.

**Decision:** Author shaders in Slang, compile to SPIR-V as a build step (not
at runtime), and cache pipelines via `VkPipelineCache`.

**Consequences:** Adds `slangc` as a build-time toolchain dependency; less
mainstream than GLSL, so tooling and debugging support is thinner.
