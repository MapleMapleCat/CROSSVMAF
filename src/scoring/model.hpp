#pragma once
#include "crossvmaf/model.hpp"
#include "metrics/adm3/adm3.hpp"
#include "metrics/cambi/cambi.hpp"
#include "metrics/motion3/motion3.hpp"
#include "metrics/speed_chroma/speed_chroma.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace crossvmaf::scoring {

struct PackedSupportVector {
    std::int64_t coefficient_raw{};
    std::array<std::int64_t, 4> feature_raw{};
};

enum class ScalingMethod : std::uint8_t {
    nearest,
    bilinear,
};

struct CambiParameters {
    std::uint32_t high_res_speedup{};
    std::int64_t visibility_luminance_threshold_raw{};
    std::int64_t maximum_raw{};
};

struct SpeedChromaParameters {
    std::int64_t kernel_scale_raw{};
    std::int64_t prescale_raw{};
    ScalingMethod prescale_method{ScalingMethod::nearest};
    std::int64_t sigma_nn_raw{};
    std::int64_t nn_floor_raw{};
    std::uint32_t weight_variance_mode{};
    std::int64_t maximum_raw{};
};

struct Adm3Parameters {
    std::int64_t dlm_weight_raw{};
    std::int64_t enhancement_gain_limit_raw{};
    std::int64_t noise_weight_raw{};
    std::int64_t normalized_view_distance_raw{};
    std::uint32_t reference_display_height{};
    std::int64_t minimum_raw{};
    std::uint32_t csf_mode{};
    std::array<std::uint32_t, 4> horizontal_csf_raw{};
    std::array<std::uint32_t, 4> diagonal_csf_raw{};
};

struct Motion3Parameters {
    std::int64_t maximum_raw{};
    bool five_frame_window{};
    bool moving_average{};
};

struct PackedModel {
    ModelVariant variant{};
    std::string_view name;
    std::string_view source_sha256;
    bool hfr{};
    CambiParameters cambi;
    SpeedChromaParameters speed_chroma;
    Adm3Parameters adm3;
    Motion3Parameters motion3;

    std::int64_t gamma_raw{};
    std::int64_t rho_raw{};
    std::int64_t output_slope_raw{};
    std::int64_t output_intercept_raw{};
    std::array<std::int64_t, 4> feature_slope_raw{};
    std::array<std::int64_t, 4> feature_intercept_raw{};

    std::int64_t transform_p0_raw{};
    std::int64_t transform_p1_raw{};
    std::int64_t transform_p2_raw{};
    std::int64_t clip_min_raw{};
    std::int64_t clip_max_raw{};
    std::int64_t chroma_correction_raw{};

    const PackedSupportVector *support_vectors{};
    std::size_t support_vector_count{};
};

[[nodiscard]] const PackedModel *find_packed_model(ModelVariant variant) noexcept;

struct ModelSnapshot {
    metrics::cambi::Config cambi;
    metrics::speed_chroma::Config speed_chroma;
    metrics::adm3::Config adm3;
    metrics::motion3::Config motion3;
};

[[nodiscard]] Result<ModelSnapshot> make_model_snapshot(ModelVariant model) noexcept;

} // namespace crossvmaf::scoring
