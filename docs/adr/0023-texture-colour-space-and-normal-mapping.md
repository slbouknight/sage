# 0023: Colour space is an explicit argument, with two fallback textures

**Status:** Accepted

**Context:** M4 hardcoded `VK_FORMAT_R8G8B8A8_SRGB` for every texture, which was
correct while base colour was the only map loaded. M5 adds normal and
metallic-roughness maps, which store measurements rather than colours. Sampling
them through an sRGB format applies a decode curve to data that was never gamma
encoded — a packed normal comes back skewed, a roughness value comes back wrong.
Nothing fails; the image just looks slightly off in a way that is easy to blame
on the BRDF.

**Decision:** `Texture` takes a `TextureColorSpace` — `srgb` or `linear` — with
**no default value**. A default is precisely how this mistake gets made, and
naming the intent at the call site (`TextureColorSpace::linear` beside a normal
map) documents the reasoning where it is needed. Base colour and emissive are
sRGB; normal and metallic-roughness are linear.

`TextureRegistry` owns two built-in textures rather than one. Slot 0 stays a 1x1
white, which is the correct identity for base colour, emissive *and*
metallic-roughness: glTF says a material with no MR texture uses its factors
directly, and `factor * 1.0` is that factor. Slot 1 is a 1x1 `(128, 128, 255)`
flat normal, because a normal map's identity is `(0, 0, 1)` and white would
decode to a normal tilted 55 degrees off every surface. That fallback is itself
linear — an sRGB decode would turn 128 into 0.216 rather than 0.502 and bias the
result it exists to keep neutral. Two fallbacks keep the shader branchless: every
material indexes a valid descriptor whether or not it has the map.

The image cache in `load_gltf` is keyed on image index *and* colour space
together, because one image wanted in both spaces needs two slots — the format
is baked into the view.

Tangent-space normals are rotated into world space by an explicit linear
combination, `sampled.x * T + sampled.y * B + sampled.z * N`, rather than
`mul(sampled, float3x3(T, B, N))`. The two are the same transform, but the matrix
form silently transposes under a convention mismatch, and `mesh.slang` already
carries a comment about one such trap from M2. A transposed TBN produces normals
that look plausible while being wrong, which is the worst thing to debug.

**Consequences:** Every `Texture` construction now states its colour space,
including the two fallbacks. `Vertex` grows to 48 bytes with glTF's `TANGENT` —
a `vec4` whose `w` is a handedness sign, not padding; dropping it lights
mirrored UV islands inside out. A primitive without `TANGENT` gets an arbitrary
default and its normal map will be wrong: generating tangents from UVs is real
work and no vendored asset needs it. Normal maps are assumed OpenGL-style, with
+Y up; a DirectX-convention map renders dents where bumps should be, and would
need its green channel flipped. `normal_scale` is now read from
`normalTexture.scale`, and setting it to zero is the cleanest A/B test for
whether normal mapping is doing anything at all.
