# 0021: Resolve glTF image URIs by hand, with a white fallback at slot 0

**Status:** Accepted

**Context:** M4's last step turns the images a glTF references into bindless
sampled-image slots. fastgltf offers `Options::LoadExternalImages`, which reads
every referenced image into memory and hands back `sources::Array` of encoded
bytes. Tested against the real case rather than assumed: with three of Lantern's
four PNGs absent, `loadGltf` returns `Error` — "An external buffer was not
found" — and the entire model fails to load. One missing texture takes the whole
asset down. It also loads eagerly: all four of Lantern's maps total roughly 9 MB
of encoded bytes, and M4 samples exactly one of them.

**Decision:** Do not enable `LoadExternalImages`. Read `sources::URI` from
`Image::data`, resolve it against the glTF's parent directory, and load each
file through `TextureRegistry`, which owns every `Texture` and hands out slots.
Every failure path — a missing file, a non-local URI, a non-zero file offset, an
embedded source, a texture whose `imageIndex` is absent because an extension
supplies it — degrades to a warning plus the fallback slot rather than an abort.
Slot 0 is always a 1x1 opaque **white** texture, not magenta: it is multiplied by
the material's `base_color_factor`, so white is the multiplicative identity and a
factor-only material renders correctly instead of announcing itself as an error.
That is also what lets the shader index a valid descriptor unconditionally —
`PARTIALLY_BOUND` makes an unwritten slot legal to leave empty but undefined to
read. Images are deduplicated by glTF image index, because several textures
routinely share one image and each decode is a full staging upload plus a mip
chain — 16 MB and eleven blits for a 2048x2048 map.

**Consequences:** GLB-embedded and base64 data-URI images are not supported;
both arrive as `sources::BufferView` or `sources::Array`, holding bytes rather
than a path, and need a `stbi_load_from_memory` path in `Texture`. That is
additive rather than a redesign, and no vendored asset needs it today. Slot
allocation lives in `TextureRegistry` rather than `BindlessSet` because the
registry is the only thing that registers images — a counter on the set would be
state with one writer and no other reader. Only `baseColorTexture` is resolved,
so Lantern's other three maps are vendored but unread until M5 wires up the
BRDF; that milestone must also stop assuming `VK_FORMAT_R8G8B8A8_SRGB`, since
normal and metallic-roughness maps store measurements rather than colours and
have to be UNORM. `TextureRegistry` holds `unique_ptr<Texture>` over a
forward-declared type, so its destructor must be defined out of line — defaulting
it in the header fails to compile, because `unique_ptr` needs a complete type
where the destructor is instantiated.
