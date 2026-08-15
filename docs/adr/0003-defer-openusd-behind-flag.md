# 0003: Defer OpenUSD behind `SAGE_ENABLE_USD` (default OFF)

**Status:** Superseded in M3 — OpenUSD is cut from the roadmap entirely.

**Context:** OpenUSD ingestion is M6 scope and pulls in a heavy, slow-to-build
dependency tree that nothing before M6 needs. Every earlier milestone builds
and iterates against glTF via fastgltf instead.

**Decision:** Gate all USD code behind `option(SAGE_ENABLE_USD ... OFF)` in
CMake so it never builds — and vcpkg never fetches it — unless explicitly
turned on.

**Consequences:** CI and local dev stay fast for M0–M5; USD integration work
can't start until the flag and its build wiring exist.

---

**Superseded in M3.** OpenUSD is removed from the roadmap and `SAGE_ENABLE_USD`
is deleted. Deciding how to load glTF forced the question of whether USD
belonged here at all, and it does not: sage exists to demonstrate Vulkan and
CUDA, while USD is a scene-description and interchange format whose value is
composition across a studio pipeline. It would have been the one milestone that
taught a reader nothing about the project's actual subject.

Two practical points reinforced it. USD geometry is *harder* to consume than
glTF, not easier — `UsdGeomMesh` yields arbitrary polygons needing
triangulation, possible subdivision surfaces, and primvars with varying
interpolation, where glTF hands over triangles and typed accessors. And USD's
`imaging` feature pulls in OpenGL, which cuts against this project's
Vulkan-only posture.

Note that this ADR only ever argued for *deferring* USD; it never argued for
including it. That absence is what made the commitment worth re-examining.
M6 becomes a CUDA simulation on the M5 interop path instead, which serves the
stated goal directly. USD work belongs in a separate, pipeline-oriented
project.
