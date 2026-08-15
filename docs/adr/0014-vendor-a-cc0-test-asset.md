# 0014: Vendor one CC0 test asset, geometry files only

**Status:** Accepted

**Context:** A renderer that cannot render anything on a fresh clone is hard to
evaluate. The options were committing a model, fetching sample assets at
configure time, or requiring the user to supply one. Fetching adds a network
dependency to configuring, which would also hit CI; requiring one means
`./sage` with no arguments has nothing to draw.

**Decision:** Vendor Khronos' **Lantern** under `assets/lantern/`, and accept a
path argument to override it. Only `Lantern.gltf` and `Lantern.bin` are
committed — 237 KB — with the texture images deliberately omitted, since
materials are deferred and the `.glb` bundle would have been 9.5 MB. Lantern was
chosen over smaller alternatives because it is CC0 rather than a bespoke licence
(Duck is under Sony's SCEA Shared Source Licence, awkward to vendor beside MIT
code), and because its four nodes — a parent carrying a rotation over three
translated children — actually exercise hierarchy flattening, which a
single-mesh model would not.

**Consequences:** The repository carries a 237 KB binary, and `.gitattributes`
marks `*.bin` and `*.glb` as binary so they are not line-ending mangled.
Attribution lives in `assets/README.md`; CC0 does not require it, but recording
provenance for vendored third-party data is worth doing regardless. The omitted
textures mean the glTF's `images` entries resolve to nothing, which is only safe
because the loader never requests image loading — restoring materials later
means re-vendoring the textures or switching to the `.glb`.
