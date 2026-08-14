struct Vertex
{
    float3 position;
    float3 color;
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
    float3 color : COLOR;
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
    output.color = vertex.color;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}