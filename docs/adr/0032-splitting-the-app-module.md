# 0032: Splitting the app module, and extracting by testability

**Status:** Accepted

**Context:** `application.cpp` had reached 2966 lines — seven times the next
largest file in the repo, `gltf_loader.cpp` at 408 — with 57 methods and about
60 data members on one class. Nothing was broken, but v2's deferred path adds a
G-buffer and a full-screen lighting pass, which is more `record_*` into the file
that was already the problem.

The worse finding was structural rather than cosmetic. `src/app` was an
`add_executable`, not a library, so **nothing in those 2966 lines could be
linked into the test binary**. The undo stack shipped with zero tests — 250-odd
lines of cursor arithmetic with fork-on-new-edit semantics — not because it was
skipped but because it was unreachable. So did the shadow-frustum fit, the
scene bounds and every screen-to-world conversion. `tests/` covered `core` and
`gpu` only, and that had been read as a gap in diligence when it was a gap in
the build.

**Decision:** Split in four passes, ordered so each one is verifiable before the
next begins.

**1 — `sage_app` becomes a static library with a thin `sage_editor` over it.**
Twenty lines of CMake, matching `sage_core` and `sage_gpu`. `OUTPUT_NAME` keeps
the binary where it was. This has no user-visible effect whatsoever and is the
only reason the other three passes are worth anything.

**2 — Extract by testability, not by subsystem.** The obvious split was "UI
module and rendering module", and it is the wrong first cut: the fifteen
tonemap, FXAA, shadow-bias and ambient fields are written by panels and read by
passes, so halving the class along that line leaves them with no home and forces
one half to hold a pointer into the other — the same coupling, one indirection
deeper. What came out first instead were the pieces that are arithmetic:
`EditHistory`, `scene_query` (bounds, light fit, light collection, child table)
and `viewport_mapping` (logical→framebuffer→NDC, the cursor ray, the ground
plane). 68 tests over the three.

**3 — `RenderSettings` first, then `Renderer`.** The settings struct is the seam
the naive split was missing, and once it exists the halves separate cleanly.
`Renderer` owns the five targets, the shadow map, the pipeline cache and all
five pipelines; everything else it needs arrives per frame in a `FrameView`
gathered at one call site. Three couplings had to be lifted out rather than
moved: `write_frame_data` was querying the graph for the first directional
light, `record_scene` was reading three editor flags to decide whether to draw
light icons, and the outline and pick passes were reading the selection and the
pending pick directly. Capture stayed behind — host readback and PNG writing is
I/O, and it touches no renderer-owned resource.

**4 — The panels move to `editor_ui.cpp` as a file split, deliberately not a
class split.** They read and write the scene, the selection, the undo history
and the pending queues; `draw_properties_panel` interleaves ImGuizmo
decompose/recompose with ImGui widget-activation state. An `EditorUi` holding an
`Application&` would be ceremony: identical coupling plus a header to keep in
step. They stay `Application` members in a second translation unit. What that
buys is real but modest — ImGui, ImGuizmo and `imgui_internal.h` no longer reach
the translation unit holding the frame loop and the scene editing.

**Consequences:** `application.cpp` is 1257 lines, down from 2966. The suite
went from 64 tests to 132.

The module is not uniformly small and this is worth stating plainly rather than
claiming a tidier outcome than was achieved: `renderer.cpp` is 894 lines and
`editor_ui.cpp` is 721. Both are cohesive — one is seven render passes, the
other is twelve panels — and both are a single subject rather than the four
unrelated ones `application.cpp` used to hold. But neither is small, and if
`renderer.cpp` grows much past this under v2 it wants splitting by pass rather
than left to become the new outlier.

Two latent problems surfaced that were not the point of the exercise:

- `viewport_texel` subtracted `ImGui::GetMainViewport()->Pos` and
  `placement_point` did not. Only `ImGuiConfigFlags_DockingEnable` is set, never
  `ViewportsEnable`, so `Pos` is always `(0,0)` and the two agreed by accident.
  They now share one conversion, with a test that would have caught the
  divergence.
- The test binary never called `core::log::init()`, because `main()` was the
  only caller. `EditHistory`'s log line aborted twelve tests the first time they
  ran. Fixed with a Catch2 listener, which is per-binary rather than per-test so
  the next extraction that logs does not rediscover it.

**On verifying a refactor that changes nothing.** The renderer half has no unit
tests and cannot easily have them: the passes need a device, a swapchain and a
bindless set. "It compiles and validation is quiet" has been weak evidence
throughout this project — the cylinder wound inside out, the shadow pointing
behind the subject and the bias in the wrong units were all invisible to both.
What was used instead: a symmetric inventory of every `Application::` and
`Renderer::` definition across the module before and after, which came back 55
and 55 with no difference; a clean rebuild on both compilers with zero warnings;
and running the editor for ten seconds with the validation layers on, which
exercised every barrier and layout transition that moved and reported nothing.
That is short of a pixel comparison — no `xdotool` on the machine to trigger a
capture — and the gap is acknowledged rather than papered over.

This supersedes the last paragraph of
[ADR 0031](0031-deletion-and-undo.md), which recorded that the undo stack "lives
on `Application`, which needs a device, so it is not reachable from a unit
test". That was true when written and is the exact limitation this change
removes: `EditHistory` now has 15 tests, including the one that matters most —
undo on an empty history is a no-op rather than an unsigned underflow that would
wrap the cursor to `SIZE_MAX` and index out of bounds on the next call.

Still untested, and honestly so: the render passes, the panels, and
`Application` itself.
