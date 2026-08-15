struct Vertex
{
    float3 position;
    float3 normal;
};

struct PushConstants
{
    Vertex* vertices;
    float4x4 mvp;
};

[vk::push_constant]
PushConstants pc;

struct VertexOutput
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
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
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target
{
    const float3 light_direction = normalize(float3(0.5, 1.0, 0.8));

    // Interpolated normals are not unit length, interpolating between two
    // unit vectors shortens the result, so renormalize per fragment.
    const float3 normal = normalize(input.normal);

    // Half-Lambert: remaps N.L from [-1, 1] to [0, 1] instead of clamping, so
    // faces turned away stay visible rather than going black. Not physically
    // correct; it is a readability choice for a viewer with a single light.
    const float diffuse = saturate(dot(normal, light_direction)) * 0.5 + 0.5;

    const float3 base_color = float3(0.8, 0.8, 0.85);

    return float4(base_color * diffuse, 1.0);
}