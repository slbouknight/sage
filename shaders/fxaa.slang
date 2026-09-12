// FXAA: fast approximate anti-aliasing, after Lottes.
//
// This is a post-process, so it never sees geometry, coverage or depth -- only
// the finished image. It finds edges by looking at where luma changes sharply,
// works out which way the edge runs, walks along it to find how far it extends
// in each direction, and then shifts the sample point perpendicular to it. The
// bilinear filter does the actual blending: a lookup offset half a texel across
// an edge returns a mix of both sides, which is what a coverage-based sampler
// would have produced, arrived at from the other end.
//
// The alternative was MSAA, which is genuinely better on geometric edges. It
// was not taken because this renderer writes an R32_UINT object-id attachment
// alongside colour, and resolving that is meaningless -- averaging two object
// ids yields a third, belonging to neither object, which would break picking
// and outlining together. Every attachment would have to be multisampled and
// the id one resolved by some custom rule. FXAA costs one pass and touches
// nothing else.
//
// It works in sRGB space, on purpose. See the note on the input binding.

// Binding 5 of set 0: the tonemapped image, with a LINEAR clamped sampler.
//
// Sampler2D, not Texture2D with Load(): the whole method depends on sampling
// *between* texel centres and letting the hardware blend. Load() would return
// one texel and there would be nothing to anti-alias with.
//
// The values it returns are sRGB-encoded, because the target is UNORM and the
// tonemap encoded by hand. That is deliberate -- luma has to be perceptual for
// "this is an edge" to mean what the eye means by it.
[vk::binding(5, 0)]
Sampler2D ldr_color;

struct PushConstants
{
    // 1 / viewport size, in pixels. Every offset below is in these units.
    float2 inverse_extent;
    // Minimum local contrast, as a fraction of the brighter side, before a
    // pixel is treated as an edge at all.
    float edge_threshold;
    // Floor on that test, so near-black regions -- where a small absolute
    // difference is a large relative one -- do not register as edges and get
    // smeared.
    float edge_threshold_min;
    // How much of the blur comes from the local sub-pixel average rather than
    // the edge walk. Higher softens single-pixel features; too high and the
    // whole image loses bite.
    float subpixel_quality;
    // 0 passes the image through untouched, which is what makes an A/B
    // comparison on one frame possible.
    int enabled;
};

[vk::push_constant]
PushConstants pc;

// Iterations of the edge walk, and how far each one steps. Growing steps cover
// long edges without paying for every texel along them; the first few stay at 1
// so short edges -- which is most of them -- are measured exactly.
static const int k_search_steps = 12;
static const float k_search_step_size[k_search_steps] = {
    1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0, 8.0
};

// Rec. 709 luma. Green dominates because the eye's response does; using an
// unweighted average would miss edges that are mostly a green change.
float luma(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float luma_at(float2 uv)
{
    return luma(ldr_color.SampleLevel(uv, 0.0).rgb);
}

// Exact inverse of the tonemap's encode. Applied on the way out because the
// swapchain is an _SRGB format and its colour write will encode again; without
// this the image would be encoded twice and come out washed out.
float3 srgb_to_linear(float3 c)
{
    const float3 low = c / 12.92;
    const float3 high = pow(max((c + 0.055) / 1.055, 0.0), 2.4);
    return lerp(low, high, step(0.04045, c));
}

// A single oversized triangle, as in the tonemap and outline passes.
[shader("vertex")]
float4 vertex_main(uint vertex_id: SV_VertexID) : SV_Position
{
    const float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

[shader("fragment")]
float4 fragment_main(float4 position: SV_Position) : SV_Target
{
    const float2 uv = position.xy * pc.inverse_extent;
    const float3 center = ldr_color.SampleLevel(uv, 0.0).rgb;

    if (pc.enabled == 0)
    {
        return float4(srgb_to_linear(center), 1.0);
    }

    // Step 1: is there an edge here at all? Four neighbours are enough to
    // answer that, and the early-out is what keeps this cheap -- most of the
    // screen is flat and leaves here.
    const float luma_center = luma(center);
    const float luma_down = luma_at(uv + float2(0.0, 1.0) * pc.inverse_extent);
    const float luma_up = luma_at(uv + float2(0.0, -1.0) * pc.inverse_extent);
    const float luma_right = luma_at(uv + float2(1.0, 0.0) * pc.inverse_extent);
    const float luma_left = luma_at(uv + float2(-1.0, 0.0) * pc.inverse_extent);

    const float luma_min = min(luma_center, min(min(luma_down, luma_up), min(luma_right, luma_left)));
    const float luma_max = max(luma_center, max(max(luma_down, luma_up), max(luma_right, luma_left)));
    const float luma_range = luma_max - luma_min;

    if (luma_range < max(pc.edge_threshold_min, luma_max * pc.edge_threshold))
    {
        return float4(srgb_to_linear(center), 1.0);
    }

    // Step 2: the corners, needed to tell a horizontal edge from a vertical
    // one. Only fetched past the early-out, which is the point of doing the
    // cheap test first.
    const float luma_down_left = luma_at(uv + float2(-1.0, 1.0) * pc.inverse_extent);
    const float luma_up_right = luma_at(uv + float2(1.0, -1.0) * pc.inverse_extent);
    const float luma_up_left = luma_at(uv + float2(-1.0, -1.0) * pc.inverse_extent);
    const float luma_down_right = luma_at(uv + float2(1.0, 1.0) * pc.inverse_extent);

    const float luma_down_up = luma_down + luma_up;
    const float luma_left_right = luma_left + luma_right;
    const float luma_left_corners = luma_down_left + luma_up_left;
    const float luma_down_corners = luma_down_left + luma_down_right;
    const float luma_right_corners = luma_down_right + luma_up_right;
    const float luma_up_corners = luma_up_right + luma_up_left;

    // Second derivative along each axis. The larger one is the direction the
    // image changes fastest across, so the edge itself runs the other way.
    const float edge_horizontal = abs(-2.0 * luma_left + luma_left_corners) +
                                  abs(-2.0 * luma_center + luma_down_up) * 2.0 +
                                  abs(-2.0 * luma_right + luma_right_corners);
    const float edge_vertical = abs(-2.0 * luma_up + luma_up_corners) +
                                abs(-2.0 * luma_center + luma_left_right) * 2.0 +
                                abs(-2.0 * luma_down + luma_down_corners);

    const bool is_horizontal = edge_horizontal >= edge_vertical;

    // Step 3: which side of this pixel the edge lies on. Whichever neighbour
    // differs more is the one across the boundary.
    const float luma_side_negative = is_horizontal ? luma_up : luma_left;
    const float luma_side_positive = is_horizontal ? luma_down : luma_right;
    const float gradient_negative = abs(luma_side_negative - luma_center);
    const float gradient_positive = abs(luma_side_positive - luma_center);

    const bool steepest_negative = gradient_negative >= gradient_positive;
    const float gradient_scaled = 0.25 * max(gradient_negative, gradient_positive);

    // Half a texel, towards the edge. This is the offset that will eventually
    // make the bilinear fetch straddle the boundary.
    float step_length = is_horizontal ? pc.inverse_extent.y : pc.inverse_extent.x;
    float luma_local_average = 0.0;
    if (steepest_negative)
    {
        step_length = -step_length;
        luma_local_average = 0.5 * (luma_side_negative + luma_center);
    }
    else
    {
        luma_local_average = 0.5 * (luma_side_positive + luma_center);
    }

    float2 current_uv = uv;
    if (is_horizontal)
    {
        current_uv.y += step_length * 0.5;
    }
    else
    {
        current_uv.x += step_length * 0.5;
    }

    // Step 4: walk along the edge in both directions until the luma stops
    // matching, which is where the edge ends. The two distances say where this
    // pixel sits along it, and that decides how much to shift.
    const float2 offset = is_horizontal ? float2(pc.inverse_extent.x, 0.0)
                                        : float2(0.0, pc.inverse_extent.y);

    float2 uv_negative = current_uv - offset * k_search_step_size[0];
    float2 uv_positive = current_uv + offset * k_search_step_size[0];

    float luma_end_negative = luma_at(uv_negative) - luma_local_average;
    float luma_end_positive = luma_at(uv_positive) - luma_local_average;

    bool reached_negative = abs(luma_end_negative) >= gradient_scaled;
    bool reached_positive = abs(luma_end_positive) >= gradient_scaled;
    bool reached_both = reached_negative && reached_positive;

    for (int i = 1; i < k_search_steps; ++i)
    {
        if (reached_both)
        {
            break;
        }
        if (!reached_negative)
        {
            uv_negative -= offset * k_search_step_size[i];
            luma_end_negative = luma_at(uv_negative) - luma_local_average;
            reached_negative = abs(luma_end_negative) >= gradient_scaled;
        }
        if (!reached_positive)
        {
            uv_positive += offset * k_search_step_size[i];
            luma_end_positive = luma_at(uv_positive) - luma_local_average;
            reached_positive = abs(luma_end_positive) >= gradient_scaled;
        }
        reached_both = reached_negative && reached_positive;
    }

    const float distance_negative =
        is_horizontal ? (uv.x - uv_negative.x) : (uv.y - uv_negative.y);
    const float distance_positive =
        is_horizontal ? (uv_positive.x - uv.x) : (uv_positive.y - uv.y);

    const bool nearer_negative = distance_negative < distance_positive;
    const float distance_nearest = min(distance_negative, distance_positive);
    const float edge_length = distance_negative + distance_positive;

    // Furthest from either end gets the largest shift; right at an end gets
    // almost none, which is what keeps corners from being rounded off.
    float pixel_offset = (-distance_nearest / max(edge_length, 1e-6)) + 0.5;

    // If the luma at the nearer end sits on the same side of the local average
    // as this pixel does, this is not really the edge we thought it was --
    // shifting would blur something that should stay sharp.
    const bool center_is_smaller = luma_center < luma_local_average;
    const float luma_end_nearest = nearer_negative ? luma_end_negative : luma_end_positive;
    const bool correct_variation = (luma_end_nearest < 0.0) != center_is_smaller;
    float final_offset = correct_variation ? pixel_offset : 0.0;

    // Step 5: sub-pixel aliasing. The edge walk handles long edges well and
    // isolated single-pixel features not at all -- a lone bright pixel has no
    // edge to walk along. This term blends in the local 3x3 average for those.
    const float luma_average =
        (1.0 / 12.0) * (2.0 * (luma_down_up + luma_left_right) + luma_left_corners + luma_right_corners);
    const float subpixel_offset_1 = saturate(abs(luma_average - luma_center) / max(luma_range, 1e-6));
    const float subpixel_offset_2 = ((-2.0 * subpixel_offset_1 + 3.0) * subpixel_offset_1) * subpixel_offset_1;
    const float subpixel_offset_final = subpixel_offset_2 * subpixel_offset_2 * pc.subpixel_quality;

    final_offset = max(final_offset, subpixel_offset_final);

    float2 final_uv = uv;
    if (is_horizontal)
    {
        final_uv.y += final_offset * step_length;
    }
    else
    {
        final_uv.x += final_offset * step_length;
    }

    // One last bilinear fetch, now straddling the edge. The blend the hardware
    // performs here is the anti-aliasing.
    const float3 result = ldr_color.SampleLevel(final_uv, 0.0).rgb;
    return float4(srgb_to_linear(result), 1.0);
}
