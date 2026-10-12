#pragma once
#include "metrics/cambi/cambi.hpp"
#include "crossvmaf/frame.hpp"
#include "metrics/cambi/c_values.hpp"
#include <cstdint>
#include <vector>

namespace crossvmaf::metrics::cambi::reference {} // namespace crossvmaf::metrics::cambi::reference

namespace crossvmaf::metrics::cambi::reference {

class Reference final {
  public:
    Reference(FrameDescriptor descriptor, Config config,
              DetailedDiagnosticsMode diagnostics_mode = DetailedDiagnosticsMode::disabled);

    [[nodiscard]] Result<Q32Value> compute(const FrameView &distorted);
    [[nodiscard]] Result<DetailedResult> compute_detailed(const FrameView &distorted);

  private:
    struct Workspace {
        explicit Workspace(std::size_t pixel_count);

        std::vector<std::uint16_t> image_primary;
        std::vector<std::uint16_t> image_secondary;
        std::vector<std::uint16_t> mask_derivative;
        std::vector<std::uint16_t> mask_primary;
        std::vector<std::uint16_t> mask_secondary;
        std::vector<CValue> c_values;
    };

    FrameDescriptor descriptor_;
    Config config_;
    DetailedDiagnosticsMode diagnostics_mode_{};
    Workspace workspace_;
};

} // namespace crossvmaf::metrics::cambi::reference
