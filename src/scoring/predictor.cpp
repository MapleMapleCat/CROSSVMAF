#include "scoring/predictor.hpp"
#include "scoring/model.hpp"
#include "model/generated_model_pack.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace crossvmaf::scoring {
namespace {

[[nodiscard]] double decode_q32(std::int64_t raw) noexcept {
    return std::ldexp(static_cast<double>(raw), -32);
}

} // namespace

const PackedModel *find_packed_model(ModelVariant variant) noexcept {
    for (const auto &model : generated::models) {
        if (model.variant == variant) return &model;
    }
    return nullptr;
}

Result<Predictor> Predictor::create(ModelVariant variant) {
    const PackedModel *const packed = find_packed_model(variant);
    if (packed == nullptr) {
        return Result<Predictor>::failure(
            {ErrorCode::unsupported_model, "model variant is not present in the model pack"});
    }
    const double output_slope = decode_q32(packed->output_slope_raw);
    if (output_slope == 0.0) {
        return Result<Predictor>::failure(
            {ErrorCode::model_error, "model output normalization has a zero slope"});
    }

    Predictor predictor;
    predictor.gamma_ = decode_q32(packed->gamma_raw);
    predictor.rho_ = decode_q32(packed->rho_raw);
    predictor.output_slope_ = output_slope;
    predictor.output_intercept_ = decode_q32(packed->output_intercept_raw);
    predictor.transform_p0_ = decode_q32(packed->transform_p0_raw);
    predictor.transform_p1_ = decode_q32(packed->transform_p1_raw);
    predictor.transform_p2_ = decode_q32(packed->transform_p2_raw);
    predictor.clip_min_ = decode_q32(packed->clip_min_raw);
    predictor.clip_max_ = decode_q32(packed->clip_max_raw);
    predictor.chroma_correction_ = decode_q32(packed->chroma_correction_raw);
    for (std::size_t feature_index = 0U; feature_index < predictor.feature_slopes_.size();
         ++feature_index) {
        predictor.feature_slopes_[feature_index] =
            decode_q32(packed->feature_slope_raw[feature_index]);
        predictor.feature_intercepts_[feature_index] =
            decode_q32(packed->feature_intercept_raw[feature_index]);
    }

    predictor.support_vectors_.reserve(packed->support_vector_count);
    for (std::size_t vector_index = 0U; vector_index < packed->support_vector_count;
         ++vector_index) {
        const PackedSupportVector &packed_support = packed->support_vectors[vector_index];
        SupportVector support_vector;
        support_vector.coefficient = decode_q32(packed_support.coefficient_raw);
        for (std::size_t feature_index = 0U; feature_index < support_vector.features.size();
             ++feature_index) {
            support_vector.features[feature_index] =
                decode_q32(packed_support.feature_raw[feature_index]);
        }
        predictor.support_vectors_.push_back(support_vector);
    }
    return Result<Predictor>::success(std::move(predictor));
}

Result<double> Predictor::predict(const FeatureVector &features) const {
    const std::array<double, 4U> input{{
        decode_q32(features.cambi.raw_q32_32),
        decode_q32(features.speed_chroma_uv.raw_q32_32),
        decode_q32(features.adm3.raw_q32_32),
        decode_q32(features.motion3.raw_q32_32),
    }};
    std::array<double, 4U> normalized{};
    for (std::size_t index = 0; index < normalized.size(); ++index) {
        normalized[index] =
            std::fma(feature_slopes_[index], input[index], feature_intercepts_[index]);
    }

    // Check the raw feature so the missing-chroma rule remains independent of
    // the host floating-point representation.
    if (features.speed_chroma_uv.raw_q32_32 == 0) {
        const double corrected = chroma_correction_ * (1.0 - input[2U]);
        normalized[1U] = std::fma(feature_slopes_[1U], corrected, feature_intercepts_[1U]);
    }

    double decision = 0.0;
    for (const SupportVector &support_vector : support_vectors_) {
        double squared_distance = 0.0;
        for (std::size_t feature_index = 0; feature_index < normalized.size(); ++feature_index) {
            const double difference =
                normalized[feature_index] - support_vector.features[feature_index];
            squared_distance = std::fma(difference, difference, squared_distance);
        }
        const double kernel = std::exp(-gamma_ * squared_distance);
        decision = std::fma(support_vector.coefficient, kernel, decision);
    }
    decision -= rho_;

    const double score = (decision - output_intercept_) / output_slope_;
    double transformed =
        std::fma(transform_p2_, score * score, std::fma(transform_p1_, score, transform_p0_));
    if (!std::isfinite(transformed)) {
        return Result<double>::failure(
            {ErrorCode::model_error,
             "floating-point model prediction produced a non-finite score"});
    }
    transformed = std::clamp(transformed, clip_min_, clip_max_);
    return Result<double>::success(transformed);
}

} // namespace crossvmaf::scoring
