# 0026: Runtime glTF loading — bump-with-rewind, serviced between frames

**Status:** Accepted

**Context:** M6's last piece is loading arbitrary glTF files at runtime instead
of the one path `main.cpp` hands over. Three things stood in the way. All three
registries bump-allocate with no free ([ADR 0011](0011-bump-suballocated-geometry-buffer.md)),
so a second load exhausts them. `MaterialRegistry::upload` replaced the whole
table from index zero, so a second load's material indices would collide with
the first's. And every failure path in the loader was a `SAGE_VERIFY` — fine
when the path was a compile-time constant, wrong the moment a person can point
the program at a file.

**Decision:** Each registry keeps bump allocation and gains `reset()`, a rewind
of the whole thing. That is all additive loading ever needed: a scene is cleared
as a unit, never a mesh at a time, so a free list would still be machinery with
no user. `MaterialRegistry::upload` becomes `append`, returning the index its
block landed at; the loader adds that base to each primitive's local
`materialIndex`. Because `Uploader::upload_to_buffer` already takes a
destination offset, only the new block is written and the registry needs no
host-side copy of the materials already resident — a cursor is the entire
change, and all three registries now have the same shape.

`load_gltf` returns `std::optional<LoadedScene>` and takes the
`MaterialRegistry` alongside the other two, so the application never threads a
material offset it does not care about. A file that cannot be read or parsed is
a logged message; a primitive that will not fit in the geometry buffer is
dropped, leaving a partial model, because a viewer that aborts over a file
choice is a bug. `MaterialRegistry` is deliberately *not* given the same
treatment: 64 bytes a material against a capacity of 1024 means overrunning it
is a reason to raise the constant. `k_geometry_capacity` goes from 4 MiB to 64.

Loads are queued by the picker and serviced at the top of the next loop
iteration, before the swapchain acquire. A load blocks on uploads, calls
`wait_idle`, and destroys images; doing that from inside the picker's own `draw`
would put it in the middle of a frame whose command buffer is already begun. The
cost is one frame of latency.

`FilePicker` is `std::filesystem` plus ImGui, in `src/app/` rather than
`sage/gpu/` — it is editor UI with no Vulkan in it. A native dialog would be an
xdg-desktop-portal round trip or a toolkit dependency for a widget that has to
do one thing.

**Consequences:** `TextureRegistry::reset` points every scene slot back at the
white fallback before destroying its images. `PARTIALLY_BOUND` only excuses
descriptors that were *never written*; one left naming a destroyed view is
dangling whether or not a shader reaches it, and that distinction is easy to
miss. It also keeps a path→slot cache so a file loaded twice is decoded once:
measured on Lantern, a repeat additive load is ~50 ms against ~530 ms cold, and
after a clear it is cold again, which is the cache being dropped correctly.
`GeometryRegistry::add_mesh` now lays out both regions before committing either,
so a mesh whose indices do not fit does not strand its vertices in the buffer.

None of this is unit-testable: every registry needs a live device, and the
picker is in the executable rather than a library. It was verified by scripting
the request sequence — additive, additive, clear, replace, bad path, replace —
through a temporary hook, and reading the material bases (0, 1, 2, then 0 again)
and node counts (5, 10, 15, then 5) out of the log.

Running that under synchronization validation turned up an unrelated
pre-existing hazard: `Uploader::upload_image` transitions every mip level to
`TRANSFER_DST_OPTIMAL` with `dstStageMask = COPY_BIT`, but levels 1 and up are
written by `vkCmdBlitImage2` at `BLIT_BIT`, leaving the transition's write
unordered against the blit's. Fixed here by widening that one mask. Ordinary
validation never mentioned it, which is the argument for running
`VK_LAYER_VALIDATE_SYNC=1` after any change to an upload path. (The
`VK_LAYER_ENABLES` spelling still works but now warns that it is deprecated.)
