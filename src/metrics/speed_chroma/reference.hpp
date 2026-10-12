#pragma once
#include "metrics/speed_chroma/speed_chroma.hpp"
#include "crossvmaf/frame.hpp"
#include "metrics/speed_chroma/reference_math.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace crossvmaf::metrics::speed_chroma::reference {

} // namespace crossvmaf::metrics::speed_chroma::reference

namespace crossvmaf::metrics::speed_chroma::reference {

namespace detail {

struct AxisSample {
    std::uint32_t first{};
    std::uint32_t second{};
    numeric::Q32_32 fraction{};
};

[[nodiscard]] Result<std::vector<std::int32_t>> preprocess_channel(const PlaneView &plane,
                                                                   const Dimensions &dimensions,
                                                                   const Config &config,
                                                                   std::uint8_t bit_depth);

} // namespace detail

class Reference final {
  public:
    Reference(FrameDescriptor descriptor, Config config);

    [[nodiscard]] Result<Q32Value> compute(const FrameView &reference,
                                           const FrameView &distorted) noexcept;
    [[nodiscard]] Result<DetailedResult> compute_detailed(const FrameView &reference,
                                                          const FrameView &distorted) noexcept;

  private:
    struct Workspace {
        Workspace() = default;
        explicit Workspace(const Dimensions &dimensions);

        std::vector<std::int32_t> normalized;
        std::vector<std::int32_t> scaled;
        std::vector<numeric::Q32_32> filter_vertical;
        std::vector<numeric::Q32_32> filter_output;
        std::vector<numeric::Q32_32> decimated;
        std::vector<std::int32_t> reference_highpass;
        std::vector<std::int32_t> distorted_highpass;
        std::vector<detail::AxisSample> horizontal_samples;
        std::vector<detail::AxisSample> vertical_samples;
        std::vector<numeric::Q32_32> independent;
        ChannelStatistics reference_statistics;
        ChannelStatistics distorted_statistics;
    };

    FrameDescriptor descriptor_;
    Config config_;
    Dimensions dimensions_{};
    Workspace workspace_;
};

} // namespace crossvmaf::metrics::speed_chroma::reference
