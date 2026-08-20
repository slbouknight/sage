struct Vertex
{
    float3 position;
    float3 normal;
    float2 uv;
};

struct PushConstants
{
    Vertex* vertices;
    float4x4 mvp;
};

[vk::push_constant]
PushConstants pc;

// Binding 1 of set 0 is BindlessSet's combined image-sampler array.
// Slang's Sampler2D lowers to OpTypeSampledImage, which is exactly what
// VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER provides.
[vk::binding(1, 0)]
Sampler2D textures[];

// Slot 0 while there is one texture; becomes a push-constant index later.
static const uint k_base_color_slot = 0;


struct VertexOutput
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv: TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertex_main(uint vertex_id : SV_VertexID)
{
    const Vertex vertex = pc.vertices[vertex_id];

    VertexOutput output;

    // mul(matrix, vector) matches glm's column-major data pushed unmodified.
    // Slang decorates the member RowMajor and lowers this to
    // OpVectorTimesMatrix; the two cancel. Don't swap operand order without retesting
    output.position = mul(pc.mvp, float4(vertex.position, 1.0));

    // Object space, not world: with no separate model matrix in the push
    // constants there is nothing to transform by. Fine while the light is defined in the same
    // space; revisit if a model matrix is ever split out.
    output.normal = vertex.normal;
    output.uv = vertex.uv;

    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target
{
    // Unlit on purpose: this isolates the texture path. Bringing shading back
    // later once camera buffer is set up
    return textures[k_base_color_slot].Sample(input.uv);
}