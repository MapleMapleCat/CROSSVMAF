#pragma once

#include "crossvmaf/error.hpp"

#include <span>

namespace crossvmaf {

enum class PoolMethod {
    mean,
    minimum,
    harmonic_mean,
};

[[nodiscard]] Result<double> pool_scores(
    std::span<const double> scores, PoolMethod method);

}  // namespace crossvmaf
