#pragma once
#include "crossvmaf/error.hpp"
#include "crossvmaf/model.hpp"
#include "crossvmaf/result.hpp"
#include <array>
#include <vector>

namespace crossvmaf::scoring {

// Preconverts the packed model once so per-frame production prediction uses
// native double arithmetic without repeatedly decoding Q32.32 constants.
class Predictor final {
  public:
    [[nodiscard]] static Result<Predictor> create(ModelVariant variant);

    [[nodiscard]] Result<double> predict(const FeatureVector &features) const;

  private:
    struct SupportVector final {
        double coefficient{};
        std::array<double, 4U> features{};
    };

    Predictor() = default;

    std::array<double, 4U> feature_slopes_{};
    std::array<double, 4U> feature_intercepts_{};
    std::vector<SupportVector> support_vectors_;
    double gamma_{};
    double rho_{};
    double output_slope_{};
    double output_intercept_{};
    double transform_p0_{};
    double transform_p1_{};
    double transform_p2_{};
    double clip_min_{};
    double clip_max_{};
    double chroma_correction_{};
};

} // namespace crossvmaf::scoring
