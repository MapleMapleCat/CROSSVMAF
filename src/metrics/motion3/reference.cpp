#include "video/frame.hpp"
#include "metrics/motion3/reference.hpp"
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <array>
#include <cstdlib>
#include <limits>

namespace crossvmaf::metrics::motion3::reference {

numeric::Q32_32 Temporal::minimum(numeric::Q32_32 lhs, numeric::Q32_32 rhs) noexcept {
    return lhs < rhs ? lhs : rhs;
}

numeric::Q32_32 Temporal::average(numeric::Q32_32 lhs, numeric::Q32_32 rhs) noexcept {
    return numeric::Q32_32::average(lhs, rhs);
}

Result<std::vector<MotionScore>> Temporal::push(std::uint64_t index, numeric::Q32_32 sad_score) {
    if (flushed_) {
        return Result<std::vector<MotionScore>>::failure(
            {ErrorCode::invalid_state, "motion temporal state is already flushed"});
    }
    if (index != score_count_) {
        return Result<std::vector<MotionScore>>::failure(
            {ErrorCode::invalid_argument,
             "motion temporal indices must be contiguous and start at zero"});
    }
    sad_score_ring_[static_cast<std::size_t>(index % sad_score_ring_.size())] = sad_score;
    ++score_count_;
    const std::uint64_t count = score_count_;
    const auto score_at = [this](std::uint64_t score_index) noexcept {
        return sad_score_ring_[static_cast<std::size_t>(score_index % sad_score_ring_.size())];
    };
    std::vector<MotionScore> output;
    output.reserve(2U);

    if (!hfr_) {
        if (count < 2U) return Result<std::vector<MotionScore>>::success({});
        const std::uint64_t output_index = count - 2U;
        const numeric::Q32_32 processed =
            output_index == 0U ? score_at(1U)
                               : minimum(score_at(output_index), score_at(output_index + 1U));
        output.push_back({output_index, Q32Value{processed.raw()}});
        return Result<std::vector<MotionScore>>::success(std::move(output));
    }

    if (count < 3U) return Result<std::vector<MotionScore>>::success({});
    if (count == 3U) {
        const numeric::Q32_32 stamp = score_at(2U);
        previous_processed_ = stamp;
        has_previous_processed_ = true;
        output.push_back({0, Q32Value{stamp.raw()}});
        output.push_back({1, Q32Value{stamp.raw()}});
        return Result<std::vector<MotionScore>>::success(std::move(output));
    }

    const std::uint64_t output_index = count - 2U;
    const numeric::Q32_32 processed =
        output_index == 2U ? score_at(output_index + 1U)
                           : minimum(score_at(output_index - 1U), score_at(output_index + 1U));
    const numeric::Q32_32 motion3 = average(processed, previous_processed_);
    previous_processed_ = processed;
    has_previous_processed_ = true;
    output.push_back({output_index, Q32Value{motion3.raw()}});
    return Result<std::vector<MotionScore>>::success(std::move(output));
}

Result<std::vector<MotionScore>> Temporal::flush() {
    if (flushed_) {
        return Result<std::vector<MotionScore>>::failure(
            {ErrorCode::invalid_state, "motion temporal flush may only be called once"});
    }
    flushed_ = true;
    const std::uint64_t count = score_count_;
    const auto score_at = [this](std::uint64_t score_index) noexcept {
        return sad_score_ring_[static_cast<std::size_t>(score_index % sad_score_ring_.size())];
    };
    std::vector<MotionScore> output;
    output.reserve(2U);
    if (count == 0U) return Result<std::vector<MotionScore>>::success({});

    if (!hfr_) {
        const numeric::Q32_32 value = count == 1U ? numeric::Q32_32{} : score_at(count - 1U);
        output.push_back({count - 1U, Q32Value{value.raw()}});
        return Result<std::vector<MotionScore>>::success(std::move(output));
    }

    if (count < 3U) {
        for (std::uint64_t index = 0U; index < count; ++index) {
            output.push_back({index, Q32Value{0}});
        }
        return Result<std::vector<MotionScore>>::success(std::move(output));
    }

    if (!has_previous_processed_) {
        return Result<std::vector<MotionScore>>::failure(
            {ErrorCode::invalid_state, "HFR motion temporal state is incomplete"});
    }
    const numeric::Q32_32 processed = score_at(count - 1U);
    const numeric::Q32_32 motion3 = average(processed, previous_processed_);
    output.push_back({count - 1U, Q32Value{motion3.raw()}});
    return Result<std::vector<MotionScore>>::success(std::move(output));
}

} // namespace crossvmaf::metrics::motion3::reference

namespace crossvmaf::metrics::motion3::reference {
namespace {

constexpr std::array<std::int32_t, 5> reference_filter{{3571, 16004, 26386, 16004, 3571}};

[[nodiscard]] int reference_mirror(int index, int size) noexcept {
    if (index < 0) return -index;
    if (index >= size) return 2 * size - index - 2;
    return index;
}

[[nodiscard]] std::int64_t reference_arithmetic_shift_right(std::int64_t value,
                                                            unsigned shift) noexcept {
    if (value >= 0) return value >> shift;
    const std::uint64_t magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1U;
    const std::uint64_t mask = (std::uint64_t{1} << shift) - 1U;
    return -static_cast<std::int64_t>((magnitude + mask) >> shift);
}

} // namespace

Reference::Reference(FrameDescriptor descriptor, bool hfr)
    : descriptor_(descriptor), hfr_(hfr), vertical_scratch_(descriptor.width), temporal_(hfr) {
    const std::size_t pixel_count = static_cast<std::size_t>(descriptor.width) * descriptor.height;
    for (auto &luma_slot : reference_ring_) {
        luma_slot.resize(pixel_count);
    }
}

void Reference::copy_luma(const PlaneView &plane,
                          std::vector<std::uint16_t> &destination) noexcept {
    for (std::uint32_t row = 0; row < descriptor_.height; ++row) {
        const std::byte *const source =
            plane.data + static_cast<std::size_t>(row) * plane.stride_bytes;
        for (std::uint32_t column = 0; column < descriptor_.width; ++column) {
            if (descriptor_.bit_depth == 8U) {
                destination[static_cast<std::size_t>(row) * descriptor_.width + column] =
                    static_cast<std::uint16_t>(source[column]);
            } else {
                const std::size_t offset = static_cast<std::size_t>(column) * 2U;
                const auto low = static_cast<std::uint16_t>(source[offset]);
                const auto high = static_cast<std::uint16_t>(source[offset + 1U]);
                destination[static_cast<std::size_t>(row) * descriptor_.width + column] =
                    static_cast<std::uint16_t>(low | (high << 8U));
            }
        }
    }
}

numeric::Q32_32 Reference::calculate_sad_score(const std::vector<std::uint16_t> &previous,
                                               const std::vector<std::uint16_t> &current) noexcept {
    const int width = static_cast<int>(descriptor_.width);
    const int height = static_cast<int>(descriptor_.height);
    std::uint64_t sad = 0;
    const std::int64_t vertical_round = std::int64_t{1} << (descriptor_.bit_depth - 1U);

    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            std::int64_t accumulator = 0;
            for (int tap = 0; tap < 5; ++tap) {
                const int source_row = reference_mirror(row - 2 + tap, height);
                const std::size_t offset =
                    static_cast<std::size_t>(source_row) * descriptor_.width +
                    static_cast<std::size_t>(column);
                const std::int32_t difference = static_cast<std::int32_t>(previous[offset]) -
                                                static_cast<std::int32_t>(current[offset]);
                accumulator +=
                    static_cast<std::int64_t>(reference_filter[static_cast<std::size_t>(tap)]) *
                    difference;
            }
            vertical_scratch_[static_cast<std::size_t>(column)] =
                static_cast<std::int32_t>(reference_arithmetic_shift_right(
                    accumulator + vertical_round, descriptor_.bit_depth));
        }

        for (int column = 0; column < width; ++column) {
            std::int64_t accumulator = 0;
            for (int tap = 0; tap < 5; ++tap) {
                const int source_column = reference_mirror(column - 2 + tap, width);
                accumulator +=
                    static_cast<std::int64_t>(reference_filter[static_cast<std::size_t>(tap)]) *
                    vertical_scratch_[static_cast<std::size_t>(source_column)];
            }
            const std::int64_t filtered =
                reference_arithmetic_shift_right(accumulator + 32768, 16U);
            sad += filtered < 0 ? static_cast<std::uint64_t>(-filtered)
                                : static_cast<std::uint64_t>(filtered);
        }
    }

    const std::uint64_t denominator = std::uint64_t{256} * descriptor_.width * descriptor_.height;
    if (sad > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        denominator > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return numeric::Q32_32::from_integer(18);
    }
    const auto score = numeric::Q32_32::from_ratio(static_cast<std::int64_t>(sad),
                                                   static_cast<std::int64_t>(denominator));
    if (!score) return numeric::Q32_32{};
    const numeric::Q32_32 maximum = numeric::Q32_32::from_integer(18);
    return score.value() > maximum ? maximum : score.value();
}

Result<std::vector<MotionScore>> Reference::push(const FrameView &reference) {
    if (reference.index != next_index_) {
        return Result<std::vector<MotionScore>>::failure(
            {ErrorCode::invalid_argument,
             "motion reference frame indices must be contiguous and start at zero"});
    }
    const auto validation = video::validate_plane_views(descriptor_, reference);
    if (!validation) return Result<std::vector<MotionScore>>::failure(validation.error());

    const std::size_t distance = hfr_ ? 2U : 1U;
    const std::size_t current_slot = static_cast<std::size_t>(next_index_ % reference_ring_.size());
    std::vector<std::uint16_t> &current = reference_ring_[current_slot];
    copy_luma(reference.planes[0], current);
    numeric::Q32_32 sad_score{};
    if (next_index_ >= distance) {
        const std::size_t previous_slot =
            static_cast<std::size_t>((next_index_ - distance) % reference_ring_.size());
        sad_score = calculate_sad_score(reference_ring_[previous_slot], current);
    }

    const auto output = temporal_.push(next_index_, sad_score);
    ++next_index_;
    return output;
}

} // namespace crossvmaf::metrics::motion3::reference
