#pragma once

#include <array>
#include <cstdint>

namespace sage::app {

// Which curve the tonemap applies. Must match the k_operator_* constants in
// shaders/tonemap.slang.
enum class TonemapOperator : std::int32_t {
    none = 0,
    reinhard = 1,
    aces = 2,
};

// Parallel to the enum above, for the combo box. An array rather than a
// function of the enum so the two orders cannot drift apart unnoticed.
inline constexpr std::array<const char*, 3> k_tonemap_names{"None (clamp)", "Reinhard",
                                                            "ACES (fitted)"};

// Everything about how a frame is shaded and presented that a person can turn.
//
// A struct of its own because it is the seam between the two halves of the
// editor: the panels write these, the render passes read them, and nothing
// else is shared between them. Without it, splitting the UI from the renderer
// leaves fifteen fields with no home and forces one half to hold a pointer to
// the other -- which is worse coupling than the single class they came from.
//
// Plain data, no invariants. The ranges that keep a value sensible belong to
// the widget that edits it, not here: a clamp in this struct would silently
// disagree with the slider's own bounds.
struct RenderSettings {
    // Replaces the 0.03 that was compiled into the shader. Without ambient
    // occlusion or IBL this is the only thing keeping unlit faces off pure
    // black, so it is the difference between "dramatic" and "half the model is
    // missing" -- which is a judgement call, hence a slider.
    float ambient_intensity = 0.03F;

    bool shadows_enabled = true;
    // Hardware depth bias, applied while rasterising the shadow map. The spec's
    // offset is `m * slopeFactor + r * constantFactor`, and the two halves are
    // in wildly different units: m is the depth slope, but r is the smallest
    // resolvable depth difference, which for a D32_SFLOAT map is around 2^-23.
    // So a slope factor of 2 is meaningful while a constant factor of 2 is
    // worth about 1e-7 -- nothing. Measured on this hardware: 1.5 was
    // indistinguishable from 0, and visible change started in the hundreds.
    // Hence the scale difference between these two defaults.
    float shadow_depth_bias = 500.0F;
    float shadow_slope_bias = 2.0F;
    // Applied at lookup time instead, along the surface normal, and measured in
    // shadow-map texels rather than world units. Texels because the frustum is
    // refitted to the scene every frame: a bias of "0.02 world units" is
    // nothing on a cathedral and four percent of a chess set, so an absolute
    // value cannot have one sensible default. A texel is the unit the error
    // actually scales with.
    float shadow_normal_bias_texels = 1.5F;
    int shadow_pcf_radius = 2;

    // Stored as int rather than TonemapOperator to match the push constant and
    // ImGui::Combo, both of which want an int to write through.
    int tonemap_operator = static_cast<int>(TonemapOperator::aces);
    // Exposure in stops, which is the unit it is reasoned about in. Converted
    // to the linear multiplier the shader wants at push time.
    float exposure_stops = 0.0F;

    // The FXAA pass runs unconditionally and passes the image through when this
    // is off, rather than being skipped. Skipping it would mean the tonemap
    // writing to a different target depending on a checkbox, and two barrier
    // paths to keep correct; a branch in the shader costs a comparison.
    bool fxaa_enabled = true;
    float fxaa_edge_threshold = 0.125F;
    float fxaa_edge_threshold_min = 0.0312F;
    float fxaa_subpixel_quality = 0.75F;
};

}  // namespace sage::app
