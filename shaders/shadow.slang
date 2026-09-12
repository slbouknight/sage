// Depth-only pass: the scene as the directional light sees it.
//
// There is no fragment entry point on purpose. Nothing here writes colour, and
// the depth value the shadow map needs is produced by the fixed-function depth
// write from SV_Position alone. The pipeline is built with `depth_only`, which
// is what stops it looking for a `fragment_main` that does not exist.
//
// Vertices come through the same buffer-device-address pointer the main pass
// uses, so a mesh is drawn here exactly as it is drawn there -- no second
// vertex layout, no position-only copy of the geometry to keep in sync.

struct Vertex
{
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv;
};

struct Light
{
    float3 position;
    float range;
    float3 direction;
    float intensity;
    float3 color;
    uint type;
};

static const uint k_max_lights = 8;

// Must match FrameData in mesh.slang and in application.cpp. Repeated rather
// than shared because slangc compiles each file on its own; the static_asserts
// on the C++ side are what actually hold the three in agreement.
struct FrameData
{
    float4x4 view_projection;
    float3 camera_position;
    uint light_count;
    Light lights[k_max_lights];
    float4x4 light_view_projection;
    float shadow_normal_bias;
    int shadow_pcf_radius;
    float shadow_texel_size;
    uint shadow_enabled;
    float ambient_intensity;
};

// The same first three members as mesh.slang's push constants, in the same
// order, so one struct on the C++ side feeds both pipelines. The members this
// pass has no use for simply go unread.
struct PushConstants
{
    Vertex* vertices;
    FrameData* frame;
    float4x4 model;
};

[vk::push_constant]
PushConstants pc;

[shader("vertex")]
float4 vertex_main(uint vertex_id: SV_VertexID) : SV_Position
{
    const Vertex vertex = pc.vertices[vertex_id];
    const float4 world_position = mul(pc.model, float4(vertex.position, 1.0));
    return mul(pc.frame.light_view_projection, world_position);
}
