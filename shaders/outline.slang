// Selection outline, drawn as a full-screen pass over the object-id attachment
// the scene pass just wrote.
//
// The rule is one line: a pixel is on the outline if it is NOT part of the
// selection but one of its neighbours IS. Testing it that way puts the line
// just outside the silhouette, so it never eats into the object itself -- the
// inverse test (inside, with an outside neighbour) would draw over the model's
// own edge pixels and look like the mesh had been trimmed.
//
// This reads ids, not depth or normals. An id discontinuity is exactly an
// object boundary, so the edge is correct by construction: no threshold to
// tune, no false edges on creases, and self-intersecting geometry outlines as
// one shape rather than several.

// Binding 2 of set 0, not the colour-texture array at binding 1: a shader's
// sampled type must match the format's numeric type, and this one is uint.
// Declared as a plain Texture2D rather than a Sampler2D because Load() takes
// integer texel coordinates and does no filtering -- which suits an id, where
// interpolating between two of them would produce a third, meaningless one.
[vk::binding(2, 0)]
Texture2D<uint> object_ids;

// Binding 0 of set 0 is BindlessSet's storage-buffer array; slot 1 within it is
// the selection flag table, one entry per scene node.
//
// A single selected id used to live in a push constant, which worked while a
// click selected exactly one mesh. Selecting a node now selects its whole
// subtree, and a subtree is not a contiguous run of indices -- so membership is
// a lookup rather than a comparison.
[vk::binding(0, 0)]
StructuredBuffer<uint> selection_flags[];

static const uint k_selection_buffer_slot = 1;

struct PushConstants
{
    // The 3D view's rect in the attachment. The outline must not be tested
    // across the edge of the viewport, or the docked panels' border would
    // count as a discontinuity and draw a line down the side of the screen.
    int2 viewport_origin;
    int2 viewport_size;
    float3 color;
    // Half-width of the line in pixels. Integer in effect -- it is a search
    // radius, not a smooth falloff.
    int thickness;
};

[vk::push_constant]
PushConstants pc;

// A single oversized triangle rather than two triangles for a quad. It covers
// the screen with no seam down the diagonal and one fewer primitive; the
// vertices land off-screen and are clipped.
[shader("vertex")]
float4 vertex_main(uint vertex_id: SV_VertexID) : SV_Position
{
    const float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Clamped so a lookup near the viewport edge re-reads the edge texel instead
// of reading a neighbouring panel's pixels.
uint id_at(int2 texel)
{
    const int2 low = pc.viewport_origin;
    const int2 high = pc.viewport_origin + pc.viewport_size - 1;
    return object_ids.Load(int3(clamp(texel, low, high), 0));
}

// Ids are node index + 1, with 0 meaning empty space -- which is never part of
// a selection, and must not index the flag table at -1.
bool selected_at(int2 texel)
{
    const uint id = id_at(texel);
    if (id == 0)
    {
        return false;
    }
    return selection_flags[k_selection_buffer_slot][id - 1] != 0;
}

[shader("fragment")]
float4 fragment_main(float4 position: SV_Position) : SV_Target
{
    const int2 texel = int2(position.xy);

    // Inside the selection already: leave the shaded surface alone. The pass is
    // skipped on the host when nothing is selected, so there is no empty-
    // selection case to test for here.
    if (!selected_at(texel))
    {
        // A square neighbourhood rather than a plus-shaped one, so diagonal
        // edges get the same line width as axis-aligned ones. At thickness 2
        // that is 24 taps, on the handful of pixels that reach this point.
        for (int dy = -pc.thickness; dy <= pc.thickness; ++dy)
        {
            for (int dx = -pc.thickness; dx <= pc.thickness; ++dx)
            {
                if (selected_at(texel + int2(dx, dy)))
                {
                    return float4(pc.color, 1.0);
                }
            }
        }
    }

    // Not on the outline. discard rather than returning a transparent pixel:
    // the blend would be a no-op, but it would still be a read-modify-write of
    // every pixel on screen.
    discard;
    return float4(0.0);
}
