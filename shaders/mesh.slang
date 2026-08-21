struct Vertex
{
    float3 position;
    float3 normal;
    float2 uv;
};

struct FrameData
{
    float4x4 view_projection;
    float3 camera_position;
};

struct PushConstants
{
    Vertex* vertices;
    FrameData* frame;
    float4x4 model;
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
    float3 world_normal : NORMAL;
    float2 uv: TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertex_main(uint vertex_id : SV_VertexID)
{
    const Vertex vertex = pc.vertices[vertex_id];

    // Split deliberately: world_position is what lighting will need, and it is
    // unrecoverable once model and view_projection are multiplied together.
    const float4 world_position = mul(pc.model, float4(vertex.position, 1.0));

    VertexOutput output;
    output.position = mul(pc.frame.view_projection, world_position);
    // mat3 of the model matrix is correct for rotation and uniform scale. Under
    // non-uniform scale this shears normals and the inverse transpose is needed.
    // That matrix does not fit the push-constant budget, so it waits for a 
    // per object buffer.
    output.world_normal = mul((float3x3)pc.model, vertex.normal);
    output.uv = vertex.uv;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target
{
    // World space, so this direction means the same thing for every object
    // regardless of how its node is rotated.
    const float3 light_direction = normalize(float3(0.5, 1.0, 0.8));

    // Interpolating between two unit vectors shortens the result, so renormalize.
    const float3 normal = normalize(input.world_normal);

    // Half-Lambert: remaps N.L from [-1, 1] to [0, 1] rather than clamping, so
    // faces turned away stay readable instead of going flat. Clamping first
    // would pin every back face to the same value and erase the falloff.
    const float diffuse = dot(normal, light_direction) * 0.5 + 0.5;

    const float4 base_color = textures[k_base_color_slot].Sample(input.uv);
    return float4(base_color.rgb * diffuse, base_color.a);
}