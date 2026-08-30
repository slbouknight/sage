# 0024: CUDA cut; sage is a Vulkan renderer

**Status:** Accepted

**Context:** sage was framed from M0 as a "Vulkan/CUDA renderer", with interop at
M6 and a CUDA simulation after it. Scoping that work made the cost concrete: a
volumetric fire and smoke showcase needs a raymarched render path sharing nothing
with the existing rasterizer, external-memory and external-semaphore plumbing, and
a grid fluid solver — four milestones with no natural stopping point before the
end. The portfolio this project serves needs Vulkan, CUDA and Unreal
demonstrated; CUDA is already covered by a separate ray tracer, and volumetrics
belong in the Unreal project, where Niagara reaches a polished result in a
fraction of the time and the domain relevance is identical.

**Decision:** Cut CUDA. sage is a Vulkan renderer, and the roadmap past M5 is
rendering and engine work: HDR presentation and tooling, shadows, IBL. Interop
was seriously considered and rejected on scope rather than value — it is the one
skill neither a standalone CUDA project nor a standalone renderer demonstrates,
and dropping it is a real loss. It lost to the fact that the surrounding
showcase, not the interop itself, was where the four milestones went.

A finish line is defined with it. A project that reaches a polished, finished
state is worth more here than a longer one held permanently at sixty percent,
and without CUDA the roadmap had no endpoint left in it.

That finish line was initially drawn at IBL, on the assumption that the
remaining work was rendering work. It was redrawn immediately afterwards, and
the second version is the real one: **v1.0 is an editor** — HDR presentation and
capture, shadows, a mutable scene graph with runtime glTF loading, and selection
with gizmos, ending at M9. The reasoning is that this project's owner is a
simulation engineer whose day-to-day is Unreal and 3D pipelines, not a graphics
programmer; tooling that resembles a real editor demonstrates more of the
relevant skill than another shading feature would, and the renderer is already
past the bar it needed to clear. IBL moved out to v2 accordingly — it is not a
"simple feature", and a constant ambient term holds the position until then.

**Consequences:** The `rhi/` question this project deferred is now closed rather
than pending; see [ADR 0001](0001-vulkan-only-no-rhi-abstraction-yet.md). The
ordering argument in [ADR 0020](0020-brdf-and-ibl-milestones.md) — which put one
milestone rather than two ahead of interop, on the grounds that interop was the
differentiator worth protecting — no longer applies, though the BRDF-before-IBL
ordering it also argued for still holds on its own merits. Ray tracing, if it
ever lands, arrives as `VK_KHR_ray_query` rather than through CUDA, which keeps
it inside the same API and the same bindless and buffer-device-address
foundations that already suit it. README and CLAUDE.md are rewritten to describe
what the project is rather than what it was going to be; leaving a roadmap in
place that describes work nobody intends to do is worse than having no roadmap.
