#include "crossvmaf/pooling.hpp"
#include <algorithm>
#include <cmath>

namespace crossvmaf {
namespace {

class CompensatedSum final {
  public:
    void add(double value) noexcept {
        const double next_sum = sum_ + value;
        if (std::abs(sum_) >= std::abs(value)) {
            correction_ += (sum_ - next_sum) + value;
        } else {
            correction_ += (value - next_sum) + sum_;
        }
        sum_ = next_sum;
    }

    [[nodiscard]] double value() const noexcept { return sum_ + correction_; }

  private:
    double sum_{};
    double correction_{};
};

[[nodiscard]] Result<double> require_finite_result(double value) {
    if (!std::isfinite(value)) {
        return Result<double>::failure(
            {ErrorCode::invalid_argument, "score pooling produced a non-finite result"});
    }
    return Result<double>::success(value);
}

} // namespace

Result<double> pool_scores(std::span<const double> scores, PoolMethod method) {
    if (scores.empty()) {
        return Result<double>::failure(
            {ErrorCode::invalid_argument, "at least one score is required for pooling"});
    }
    for (const double score : scores) {
        if (!std::isfinite(score)) {
            return Result<double>::failure(
                {ErrorCode::invalid_argument, "pooling requires every score to be finite"});
        }
    }

    if (method == PoolMethod::minimum) {
        return Result<double>::success(*std::min_element(scores.begin(), scores.end()));
    }

    CompensatedSum sum;
    if (method == PoolMethod::mean) {
        for (const double score : scores)
            sum.add(score);
    } else if (method == PoolMethod::harmonic_mean) {
        for (const double score : scores) {
            const double denominator = score + 1.0;
            if (denominator <= 0.0) {
                return Result<double>::failure(
                    {ErrorCode::invalid_argument,
                     "harmonic pooling requires every score to be greater than -1"});
            }
            sum.add(1.0 / denominator);
        }
    } else {
        return Result<double>::failure({ErrorCode::invalid_argument, "unsupported pooling method"});
    }

    const double count = static_cast<double>(scores.size());
    const double pooled =
        method == PoolMethod::mean ? sum.value() / count : count / sum.value() - 1.0;
    return require_finite_result(pooled);
}

} // namespace crossvmaf
