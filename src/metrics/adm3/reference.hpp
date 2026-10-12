#pragma once
#include "metrics/adm3/adm3.hpp"
#include "crossvmaf/frame.hpp"
#include "crossvmaf/error.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace crossvmaf::metrics::adm3::reference {} // namespace crossvmaf::metrics::adm3::reference

namespace crossvmaf::metrics::adm3::reference {

class Reference final {
  public:
    Reference(FrameDescriptor descriptor, Config config);

    [[nodiscard]] Result<Q32Value> compute(const FrameView &reference,
                                           const FrameView &distorted) noexcept;
    [[nodiscard]] Result<DetailedResult> compute_detailed(const FrameView &reference,
                                                          const FrameView &distorted) noexcept;

  private:
    struct Workspace {
        explicit Workspace(std::size_t maximum_band_count);

        std::array<std::vector<std::int32_t>, 2> reference_low;
        std::array<std::vector<std::int32_t>, 2> distorted_low;
        std::array<std::vector<std::int32_t>, 3> reference_detail;
        std::array<std::vector<std::int32_t>, 3> restored_detail;
        std::array<std::vector<std::int32_t>, 3> additive_detail;
    };

    FrameDescriptor descriptor_;
    Config config_;
    Workspace workspace_;
};

} // namespace crossvmaf::metrics::adm3::reference
