#pragma once
#include "crossvmaf/value.hpp"
#include <optional>
#include <vector>
#include "numeric/q32.hpp"
#include <array>
#include <cstdint>

namespace crossvmaf::metrics::speed_chroma {

enum class PrescaleMethod : std::uint8_t { nearest, bilinear };

inline constexpr std::uint32_t block_size = 5U;
inline constexpr std::uint32_t elements_per_block = block_size * block_size;

struct Config {
    numeric::Q32_32 kernel_scale{numeric::Q32_32::from_integer(1)};
    numeric::Q32_32 prescale{numeric::Q32_32::from_integer(1)};
    PrescaleMethod prescale_method{PrescaleMethod::nearest};
    numeric::Q32_32 sigma_nn{};
    numeric::Q32_32 nn_floor{};
    std::uint32_t weight_variance_mode{5U};
    numeric::Q32_32 maximum{numeric::Q32_32::from_integer(45)};
};

struct Dimensions {
    std::uint32_t original_width{};
    std::uint32_t original_height{};
    std::uint32_t scaled_width{};
    std::uint32_t scaled_height{};
    std::uint32_t operating_width{};
    std::uint32_t operating_height{};
    std::uint32_t truncated_width{};
    std::uint32_t truncated_height{};
    std::uint32_t blocks_horizontal{};
    std::uint32_t blocks_vertical{};
    std::uint32_t block_count{};
    std::uint32_t submatrix_width{};
    std::uint32_t submatrix_height{};
    std::uint32_t covariance_sample_count{};
};

enum class EstimationStatus : std::uint8_t {
    computed,
    guaranteed_singular,
};

struct ParameterDiagnostics {
    EstimationStatus status{EstimationStatus::computed};
    std::uint32_t covariance_sample_count{};
    std::vector<numeric::Q32_32> means;
    std::vector<numeric::Q32_32> covariance;
    std::vector<numeric::Q32_32> tridiagonal_diagonal;
    std::vector<numeric::Q32_32> tridiagonal_subdiagonal;
    std::uint32_t negative_pivot_count{};
    std::uint32_t stabilized_pivot_count{};
};

struct ChannelDiagnostics {
    ParameterDiagnostics reference_parameters;
    ParameterDiagnostics distorted_parameters;
    std::vector<numeric::Q32_32> reference_entropies;
    std::vector<numeric::Q32_32> reference_variances;
    std::vector<numeric::Q32_32> distorted_entropies;
    std::vector<numeric::Q32_32> distorted_variances;
};

struct DetailedDiagnostics {
    ChannelDiagnostics u;
    ChannelDiagnostics v;
};

struct DetailedResult {
    Q32Value u{};
    Q32Value v{};
    Q32Value uv{};
    bool u_singular{};
    bool v_singular{};
    Dimensions dimensions{};
};

enum class CompactFactorStatus : std::uint8_t {
    computed,
    singular,
    arithmetic_failure,
};

struct CompactFactorDiagnostics {
    CompactFactorStatus status{CompactFactorStatus::arithmetic_failure};
    bool factorization_payload_available{};
    bool regularity_payload_available{};
    bool ldl_payload_available{};
    std::int32_t matrix_exponent{};
    std::uint32_t stream_status_flags{};
    bool singular{};
    std::uint32_t sign_variation_count{};
    std::uint32_t exact_zero_count{};
    std::array<std::int64_t, elements_per_block - 2U> reflector_tau_raw{};
    std::array<std::int64_t, (elements_per_block - 1U) * (elements_per_block - 2U) / 2U>
        reflector_tails_raw{};
    std::array<std::int64_t, elements_per_block> diagonal_raw{};
    std::array<std::int64_t, elements_per_block - 1U> subdiagonal_raw{};
    std::array<std::int64_t, elements_per_block> diagnostic_diagonal_raw{};
    std::array<std::int64_t, elements_per_block - 1U> diagnostic_subdiagonal_raw{};
    std::array<std::int64_t, elements_per_block> ldl_pivots_raw{};
    std::array<std::int64_t, elements_per_block - 1U> ldl_multipliers_raw{};
};

[[nodiscard]] constexpr bool guarantees_singular_covariance(const Dimensions &dimensions) noexcept {
    return dimensions.covariance_sample_count <= elements_per_block;
}

} // namespace crossvmaf::metrics::speed_chroma

namespace crossvmaf::metrics::speed_chroma::reference {
using metrics::speed_chroma::block_size;
using metrics::speed_chroma::ChannelDiagnostics;
using metrics::speed_chroma::CompactFactorDiagnostics;
using metrics::speed_chroma::CompactFactorStatus;
using metrics::speed_chroma::Config;
using metrics::speed_chroma::DetailedDiagnostics;
using metrics::speed_chroma::DetailedResult;
using metrics::speed_chroma::Dimensions;
using metrics::speed_chroma::elements_per_block;
using metrics::speed_chroma::EstimationStatus;
using metrics::speed_chroma::guarantees_singular_covariance;
using metrics::speed_chroma::ParameterDiagnostics;
using metrics::speed_chroma::PrescaleMethod;
} // namespace crossvmaf::metrics::speed_chroma::reference

namespace crossvmaf::metrics::speed_chroma::vulkan {
using metrics::speed_chroma::block_size;
using metrics::speed_chroma::ChannelDiagnostics;
using metrics::speed_chroma::CompactFactorDiagnostics;
using metrics::speed_chroma::CompactFactorStatus;
using metrics::speed_chroma::Config;
using metrics::speed_chroma::DetailedDiagnostics;
using metrics::speed_chroma::DetailedResult;
using metrics::speed_chroma::Dimensions;
using metrics::speed_chroma::elements_per_block;
using metrics::speed_chroma::EstimationStatus;
using metrics::speed_chroma::guarantees_singular_covariance;
using metrics::speed_chroma::ParameterDiagnostics;
using metrics::speed_chroma::PrescaleMethod;
} // namespace crossvmaf::metrics::speed_chroma::vulkan
