# 0013: glTF loading, geometry only, flattened at load time

**Status:** Accepted

**Context:** M3.2 needs to turn a glTF file into draws. glTF stores attributes
in accessors over buffer views with varying component types, strides, and
optional interleaving; indices may be 16- or 32-bit; and geometry sits under a
node hierarchy whose transforms compose. Materials, textures, animations,
cameras and skins are all in the format too, and none of them are needed to
prove the asset pipeline works.

**Decision:** `load_gltf` reads positions, normals and indices only, and returns
a flat `Scene` of `(MeshView, world transform)` pairs plus world-space bounds.
The node hierarchy is resolved once at load time via
`fastgltf::iterateSceneNodes` rather than walked per frame. `copyFromAccessor`
normalises component types, so indices always reach the registry as `uint32`.
`GenerateMeshIndices` synthesises indices for non-indexed primitives so the rest
of the pipeline can assume they exist. Images are never requested, so a glTF
referencing textures that were not vendored still loads. Primitives that are
non-triangle or lack positions are skipped with a warning; an empty scene is
fatal.

**Consequences:** Everything renders untextured grey under a single hardcoded
light — the bindless sampled-image array stays empty until materials land. The
flattened scene cannot be edited: the editor's gizmo work needs a persistent scene graph
alongside this, not a mutation of it. The returned bounds exist so an arbitrary
model can be framed automatically, since a viewer that drops the camera inside
the geometry is not usable. No fastgltf type appears in the public header, so a
second format would plug into the same `Scene` shape.
