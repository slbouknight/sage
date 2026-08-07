struct Vertex
{
    float2 position;
};

struct PushConstants
{
    Vertex* vertices;
    uint color_index;
};

[vk::push_constant]
PushConstants pc;

// Unbounded array: the size comes from the descriptor set layout, not here.
// Set 0 is one bindless set; binding 0 is its storage-buffer array.
[[vk::binding(0, 0)]]
StructuredBuffer<float4> colors[];

struct VertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR;
};

[shader("vertex")]
VertexOutput vertex_main(uint vertex_id : SV_VertexID)
{
    VertexOutput output;
    output.position = float4(pc.vertices[vertex_id].position, 0.0, 1.0);
    output.color = colors[pc.color_index][vertex_id].rgb;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}