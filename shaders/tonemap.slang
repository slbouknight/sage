// Resolves the HDR colour target to the swapchain.
//
// Shading has produced open-ended values since M5 -- a specular highlight under
// a light of intensity 2 passes 1.0 without trying -- and until M8 those were
// written straight to an 8-bit attachment, which clamped them at the hardware.
// Two highlights of very different energy came out the same flat white, and no
// amount of correctness in the BRDF could show through that. This pass is where
// the range collapses instead, deliberately and with the whole pixel in hand.
//
// The output is LINEAR. The swapchain is a _SRGB format, so the colour write
// applies the linear -> sRGB transfer function in fixed function; encoding here
// as well would apply it twice and wash the image out.

// Binding 3 of set 0. Declared Texture2D rather than Sampler2D and read with
// Load() because this pass is exactly one texel per pixel -- the target is
// swapchain-sized -- so there is nothing for a filter to interpolate. A bloom
// or a resolution-independent resolve would want a sampler; neither exists yet.
[vk::binding(3, 0)]
Texture2D<float4> hdr_color;

// Curves, selectable at runtime so two can be compared on the same frame rather
// than across a rebuild. Kept in sync with TonemapOperator in application.cpp.
static const int k_operator_none = 0;
static const int k_operator_reinhard = 1;
static const int k_operator_aces = 2;

struct PushConstants
{
    // Linear multiplier applied before the curve. The UI exposes it in stops
    // and converts, because a stop is the unit the number is actually reasoned
    // about in -- "one stop brighter" rather than "times 2".
    float exposure;
    int tonemap_operator;
};

[vk::push_constant]
PushConstants pc;

// A single oversized triangle, same as the outline pass: no vertex buffer, no
// seam down the diagonal, one primitive instead of two.
[shader("vertex")]
float4 vertex_main(uint vertex_id: SV_VertexID) : SV_Position
{
    const float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

// c / (1 + c). Maps [0, inf) onto [0, 1) with nothing ever quite reaching white,
// which is the point -- but it is a per-channel curve, so a saturated bright
// colour desaturates toward white faster than a neutral one of the same energy.
float3 tonemap_reinhard(float3 color)
{
    return color / (1.0 + color);
}

// Narkowicz's curve fit to the ACES filmic response. Not the full ACES
// pipeline: that transforms into and out of the AP1 primaries around an RRT and
// an ODT, where this is a rational approximation of the whole chain evaluated in
// the working space. Cheap, and it keeps the toe and shoulder that make
// highlights roll off rather than clip.
float3 tonemap_aces(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

[shader("fragment")]
float4 fragment_main(float4 position: SV_Position) : SV_Target
{
    const int2 texel = int2(position.xy);
    float3 color = hdr_color.Load(int3(texel, 0)).rgb;

    color *= pc.exposure;

    switch (pc.tonemap_operator)
    {
    case k_operator_reinhard:
        color = tonemap_reinhard(color);
        break;
    case k_operator_aces:
        color = tonemap_aces(color);
        break;
    default:
        // The pre-M8 behaviour, kept as an option rather than deleted: it is
        // what the hardware did at the attachment when the scene rendered
        // straight to the swapchain, so selecting it shows exactly what the
        // tonemap is buying.
        color = saturate(color);
        break;
    }

    // Alpha 1: the swapchain is composited opaque, and the HDR target's alpha
    // carries nothing meaningful -- the scene pass never writes it with intent.
    return float4(color, 1.0);
}
