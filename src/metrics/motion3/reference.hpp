#pragma once
#include "crossvmaf/error.hpp"
#include "numeric/q32.hpp"
#include "metrics/motion3/motion3.hpp"
#include <array>
#include <cstdint>
#include <vector>
#include "crossvmaf/frame.hpp"

namespace crossvmaf::metrics::motion3::reference {

class Temporal final {
  public:
    explicit Temporal(bool hfr) noexcept : hfr_(hfr) {}

    [[nodiscard]] Result<std::vector<MotionScore>> push(std::uint64_t index,
                                                        numeric::Q32_32 sad_score);
    [[nodiscard]] Result<std::vector<MotionScore>> flush();

  private:
    [[nodiscard]] static numeric::Q32_32 minimum(numeric::Q32_32 lhs, numeric::Q32_32 rhs) noexcept;
    [[nodiscard]] static numeric::Q32_32 average(numeric::Q32_32 lhs, numeric::Q32_32 rhs) noexcept;

    bool hfr_{};
    bool flushed_{};
    std::array<numeric::Q32_32, 3> sad_score_ring_{};
    std::uint64_t score_count_{};
    numeric::Q32_32 previous_processed_{};
    bool has_previous_processed_{};
};

} // namespace crossvmaf::metrics::motion3::reference

namespace crossvmaf::metrics::motion3::reference {

} // namespace crossvmaf::metrics::motion3::reference

namespace crossvmaf::metrics::motion3::reference {

class Reference final {
  public:
    Reference(FrameDescriptor descriptor, bool hfr);

    [[nodiscard]] Result<std::vector<MotionScore>> push(const FrameView &reference);
    [[nodiscard]] Result<std::vector<MotionScore>> flush() { return temporal_.flush(); }

  private:
    void copy_luma(const PlaneView &plane, std::vector<std::uint16_t> &destination) noexcept;
    [[nodiscard]] numeric::Q32_32
    calculate_sad_score(const std::vector<std::uint16_t> &previous,
                        const std::vector<std::uint16_t> &current) noexcept;

    FrameDescriptor descriptor_;
    bool hfr_{};
    std::uint64_t next_index_{};
    std::array<std::vector<std::uint16_t>, 3> reference_ring_;
    std::vector<std::int32_t> vertical_scratch_;
    Temporal temporal_;
};

} // namespace crossvmaf::metrics::motion3::reference
