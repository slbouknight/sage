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

static const uint k_light_directional = 0;
static const uint k_light_point = 1;
static const uint k_max_lights = 8;

struct FrameData
{
    float4x4 view_projection;
    float3 camera_position;
    uint light_count;
    Light lights[k_max_lights];
    // Shadows cover the first directional light only. A second shadowed light
    // needs a second map and a second depth pass, which is a scene-complexity
    // decision rather than a shader one.
    float4x4 light_view_projection;
    // World-space distance to push a sample point along its normal before
    // projecting into light space. Works where a pure depth bias struggles:
    // depth bias is measured perpendicular to the light, so on a surface nearly
    // edge-on to it the offset needed grows without bound, while moving along
    // the normal is the direction the error actually lies in.
    float shadow_normal_bias;
    // Half-width of the PCF kernel, in shadow-map texels. 0 is a single tap.
    int shadow_pcf_radius;
    // 1 / resolution. Passed rather than derived so the shader never needs to
    // know how big the map is.
    float shadow_texel_size;
    uint shadow_enabled;
    // Flat ambient. The only thing between an unlit face and pure black while
    // there is no IBL and no occlusion term, so how much of it there is is a
    // look decision rather than a constant worth compiling in.
    float ambient_intensity;
};

struct Material
{
    float4 base_color_factor;
    float3 emissive_factor;
    float metallic;
    float roughness;
    uint base_color_texture;
    uint normal_texture;
    uint metallic_roughness_texture;
    uint emissive_texture;
    float normal_scale;
    float pad0;
    float pad1;
};

// Binding 0 of set 0 is BindlessSet's storage-buffer array. The outer index
// names the material table; the inner one names a material within it.
[vk::binding(0, 0)]
StructuredBuffer<Material> materials[];

static const uint k_material_buffer_slot = 0;

struct PushConstants
{
    Vertex* vertices;
    FrameData* frame;
    float4x4 model;
    uint material_index;
    // Node index + 1, written to the object-id attachment. Zero is reserved
    // for "nothing here", which is what the attachment clears to.
    uint object_id;
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

// Binding 4 of set 0: the directional light's shadow map, paired with a
// comparison sampler. Sampler2DShadow rather than Sampler2D -- the extra
// coordinate is the depth to test against, and what comes back is not a stored
// depth but the filtered result of the test, so a LINEAR filter blends four
// lit/shadowed answers instead of four depths. Averaging depths would compare
// against a surface that exists nowhere in the scene.
[vk::binding(4, 0)]
Sampler2DShadow shadow_map;


struct VertexOutput
{
    float4 position : SV_Position;
    float3 world_position : POSITION;
    float3 world_normal : NORMAL;
    float4 world_tangent : TANGENT;
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
    output.world_position = world_position.xyz;
    // mat3 of the model matrix is correct for rotation and uniform scale. Under
    // non-uniform scale this shears normals and the inverse transpose is needed.
    // That matrix does not fit the push-constant budget, so it waits for a 
    // per object buffer.
    output.world_normal = mul((float3x3)pc.model, vertex.normal);
    output.world_tangent = float4(mul((float3x3)pc.model, vertex.tangent.xyz), vertex.tangent.w);
    output.uv = vertex.uv;
    return output;
}

static const float k_pi = 3.14159265359;

// GGX / Trowbridge-Reitz: what fraction of microfacets are oriented to bounce
// the view ray towards this light. Squaring roughness is Disney's remap, which
// makes a linear roughness slider feel linear instead of bunching near zero.
float distribution_ggx(float n_dot_h, float roughness)
{
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float denominator = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(k_pi * denominator * denominator, 1e-7);
}

float geometry_schlick_ggx(float n_dot_x, float k)
{
    return n_dot_x / max(n_dot_x * (1.0 - k) + k, 1e-7);
}

// Microfacets shadow and mask each other at grazing angles; without this the
// surface gains energy as it turns edge-on. The (r+1)^2/8 remap is the *direct
// lighting* one -- IBL prefiltering uses k = r^2/2, which is why M7 must not
// reuse this function as-is.
float geometry_smith(float n_dot_v, float n_dot_l, float roughness)
{
    const float r = roughness + 1.0;
    const float k = (r * r) / 8.0;
    return geometry_schlick_ggx(n_dot_v, k) * geometry_schlick_ggx(n_dot_l, k);
}

// Reflectance rises towards 100% at grazing angles, for every material. This is
// why a rough wooden floor still glares when you look along it.
float3 fresnel_schlick(float v_dot_h, float3 f0)
{
    return f0 + (float3(1.0) - f0) * pow(saturate(1.0 - v_dot_h), 5.0);
}

// Cook-Torrance for a single light, including the N.L cosine but excluding the
// light's own radiance -- the caller multiplies that in, so the same function
// serves a forward loop, a deferred pass, or a hit shader unchanged.
float3 evaluate_brdf(float3 normal, float3 view, float3 light_dir, float3 albedo,
                     float metallic, float roughness)
{
    const float3 half_vector = normalize(view + light_dir);

    const float n_dot_v = max(dot(normal, view), 1e-4);
    const float n_dot_l = saturate(dot(normal, light_dir));
    const float n_dot_h = saturate(dot(normal, half_vector));
    const float v_dot_h = saturate(dot(view, half_vector));

    // Dielectrics reflect about 4% head-on; metals use their base colour as F0
    // and have no diffuse at all. That one lerp is the whole metallic workflow.
    const float3 f0 = lerp(float3(0.04), albedo, metallic);

    const float d = distribution_ggx(n_dot_h, roughness);
    const float g = geometry_smith(n_dot_v, n_dot_l, roughness);
    const float3 f = fresnel_schlick(v_dot_h, f0);

    // The 4 * N.V * N.L denominator is Cook-Torrance's normalisation.
    const float3 specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);

    // Energy conservation: light Fresnel reflects cannot also refract into the
    // surface and scatter back out, and metals refract nothing.
    const float3 k_diffuse = (float3(1.0) - f) * (1.0 - metallic);

    // Lambert is albedo/pi, not albedo -- the pi is what makes it integrate to
    // albedo over the hemisphere rather than to pi times albedo.
    return (k_diffuse * albedo / k_pi + specular) * n_dot_l;
}

float3 apply_normal_map(float3 world_normal, float4 world_tangent, uint normal_texture,
                        float normal_scale, float2 uv)
{
    const float3 n = normalize(world_normal);

    // Gram-Schmidt. Interpolating across a triangle leaves T no longer exactly
    // perpendicular to N, and a skewed basis tilts every mapped normal.
    const float3 t = normalize(world_tangent.xyz - n * dot(n, world_tangent.xyz));
    // The w sign is what makes mirrored UV islands map correctly; drop it and
    // half of a symmetric model lights inside out.
    const float3 b = cross(n, t) * world_tangent.w;

    float3 sampled = textures[normal_texture].Sample(uv).xyz * 2.0 - 1.0;
    sampled.xy *= normal_scale;

    // Written out rather than mul(v, float3x3(t, b, n)): this is exactly that
    // transform, but it cannot be silently transposed by a convention mismatch.
    return normalize(sampled.x * t + sampled.y * b + sampled.z * n);
}

// How much of the directional light reaches this point: 1 fully lit, 0 fully
// shadowed, fractions in between from filtering.
float shadow_factor(float3 world_position, float3 normal)
{
    if (pc.frame.shadow_enabled == 0)
    {
        return 1.0;
    }

    const float3 biased = world_position + normal * pc.frame.shadow_normal_bias;
    const float4 light_space = mul(pc.frame.light_view_projection, float4(biased, 1.0));

    // The divide is a no-op for an orthographic projection, where w is always
    // 1. Kept so this function does not silently become wrong the day a
    // spotlight with a perspective projection wants it.
    const float3 ndc = light_space.xyz / light_space.w;

    // NDC xy is [-1, 1] and texture coordinates are [0, 1]. No Y negation here:
    // ortho_vk already applied Vulkan's flip when the map was rasterised, so
    // both sides of this lookup share one convention.
    const float2 uv = ndc.xy * 0.5 + 0.5;

    // Beyond the far plane of the light's frustum there is nothing recorded to
    // be occluded by. The sampler's opaque-white border handles xy outside the
    // map; z has no border to fall back on.
    if (ndc.z > 1.0)
    {
        return 1.0;
    }

    // A square kernel of comparison taps. Each one is already bilinear across
    // four texels thanks to the comparison sampler, so radius 1 is a 3x3 grid
    // of 2x2 filtered tests -- considerably softer than nine hard ones.
    const int radius = pc.frame.shadow_pcf_radius;
    const float texel = pc.frame.shadow_texel_size;

    float sum = 0.0;
    int taps = 0;
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            const float2 offset = float2(x, y) * texel;
            // LevelZero, not plain SampleCmp: the map has one mip, and this
            // lookup sits inside a branch, where the implicit derivatives a
            // mip selection would need are not well defined.
            sum += shadow_map.SampleCmpLevelZero(uv + offset, ndc.z);
            ++taps;
        }
    }
    return sum / float(taps);
}

// Two render targets: shaded colour, and the object id that lets the cursor
// and the outline pass work out what a pixel belongs to.
struct FragmentOutput
{
    float4 color : SV_Target0;
    uint object_id : SV_Target1;
};

[shader("fragment")]
FragmentOutput fragment_main(VertexOutput input)
{
    const Material material = materials[k_material_buffer_slot][pc.material_index];
    const float4 base_color = textures[material.base_color_texture].Sample(input.uv) * material.base_color_factor;

    // Interpolating between two unit vectors shortens the result, so renormalize.
    const float3 normal = apply_normal_map(input.world_normal, input.world_tangent,
                                           material.normal_texture, material.normal_scale,
                                           input.uv);
    const float3 view = normalize(pc.frame.camera_position - input.world_position);
    const float3 albedo = base_color.rgb;

    // glTF packs roughness in G and metallic in B, and multiplies by the
    // factors. Getting the channels backwards makes metal look rough and rough
    // things look metallic -- plausible enough to survive a casual look.
    const float3 metallic_roughness =
        textures[material.metallic_roughness_texture].Sample(input.uv).rgb;
    const float roughness = saturate(material.roughness * metallic_roughness.g);
    const float metallic = saturate(material.metallic * metallic_roughness.b);

    // Flat ambient so faces with no light on them are not pure black.
    // Replace with an irradiance probe in future IBL work.
    float3 lit = base_color.rgb * pc.frame.ambient_intensity;

    // The shadow map is fitted to one directional light, so only the first one
    // consumes it. Later directional lights are treated as fill and stay
    // unshadowed rather than reusing a map fitted to a different direction.
    bool shadow_spent = false;

    for (uint i=0; i < pc.frame.light_count; ++i)
    {
        const Light light = pc.frame.lights[i];

        float3 to_light;
        float attenuation = 1.0;

        if (light.type == k_light_point)
        {
            const float3 offset = light.position - input.world_position;
            const float distance = length(offset);
            to_light = offset / max(distance * distance, 1e-4);
            attenuation = 1.0 / max(distance * distance, 1e-4);
            if (light.range > 0.0)
            {
                // Windowed inverse-square: pure 1/d^2 never reaches zero,
                // so a light would keep costing work and leaking light
                const float window = saturate(1.0 - pow(distance / light.range, 4.0));
                attenuation = window * window;
            }
        }
        else
        {
            to_light = -normalize(light.direction);
            if (!shadow_spent)
            {
                // The geometric normal, not the mapped one: a normal map's
                // detail has no bearing on whether the light is occluded, and
                // offsetting along it pushes the sample sideways into
                // neighbouring texels for no gain.
                attenuation *= shadow_factor(input.world_position, normalize(input.world_normal));
                shadow_spent = true;
            }
        }

        const float3 radiance = light.color * (light.intensity * attenuation);
        lit += evaluate_brdf(normal, view, to_light, albedo, metallic, roughness) * radiance;
    }

        // Emissive is added, not lit -- it leaves the surface regardless of what
    // reaches it, which is why the lantern glass glows with no light nearby.
    const float3 emissive =
        textures[material.emissive_texture].Sample(input.uv).rgb * material.emissive_factor;
    lit += emissive;

    FragmentOutput output;
    output.color = float4(lit, base_color.a);
    output.object_id = pc.object_id;
    return output;
}