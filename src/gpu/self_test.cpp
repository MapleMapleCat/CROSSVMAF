#include "gpu/device.hpp"
#include "gpu/buffer.hpp"
#include "gpu/command.hpp"
#include "gpu/pipeline.hpp"
#include "numeric/rounding.hpp"
#include "numeric/q32.hpp"
#include "numeric/uint128.hpp"
#include "shaders/self_test_spv.hpp"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <sstream>
#include <utility>

namespace crossvmaf::gpu {
namespace {

constexpr std::size_t case_count = 96U;
constexpr std::size_t words_per_case = 12U;

struct ArithmeticCase {
    std::uint32_t operation{};
    std::int64_t lhs{};
    std::int64_t rhs{};
    std::int64_t expected{};
    bool success{};
    std::uint32_t prepared_denominator{};
};

struct Q48DivideResult {
    std::int64_t raw{};
    bool success{};
};

[[nodiscard]] Q48DivideResult divide_q48_rne(std::int64_t lhs, std::int64_t rhs) noexcept {
    const std::uint64_t denominator = numeric::detail::magnitude(rhs);
    if (denominator == 0U) return {};

    const bool negative = (lhs < 0) != (rhs < 0);
    const std::uint64_t value = numeric::detail::magnitude(lhs);
    const numeric::detail::UInt128 numerator{
        value >> 16U,
        value << 48U,
    };
    std::uint64_t remainder = 0U;
    bool overflow = false;
    std::uint64_t quotient =
        numeric::detail::divide_u128_u64(numerator, denominator, &remainder, &overflow);
    if (!overflow) {
        const std::uint64_t complement = denominator - remainder;
        const bool round_up =
            remainder > complement || (remainder == complement && (quotient & 1U) != 0U);
        if (round_up) {
            if (quotient == std::numeric_limits<std::uint64_t>::max()) {
                overflow = true;
            } else {
                ++quotient;
            }
        }
    }

    constexpr std::uint64_t positive_limit =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    constexpr std::uint64_t negative_limit = positive_limit + 1U;
    std::int64_t output;
    if (overflow || (!negative && quotient > positive_limit)) {
        output = negative ? std::numeric_limits<std::int64_t>::min()
                          : std::numeric_limits<std::int64_t>::max();
    } else if (negative) {
        output = quotient >= negative_limit ? std::numeric_limits<std::int64_t>::min()
                                            : -static_cast<std::int64_t>(quotient);
    } else {
        output = static_cast<std::int64_t>(quotient);
    }
    return {output, !overflow};
}

void store_raw(std::array<std::uint32_t, case_count * words_per_case> &words, std::size_t offset,
               std::int64_t raw) noexcept {
    const auto bits = static_cast<std::uint64_t>(raw);
    words[offset] = static_cast<std::uint32_t>(bits);
    words[offset + 1U] = static_cast<std::uint32_t>(bits >> 32U);
}

[[nodiscard]] std::int64_t
load_raw(const std::array<std::uint32_t, case_count * words_per_case> &words,
         std::size_t offset) noexcept {
    const std::uint64_t bits = static_cast<std::uint64_t>(words[offset]) |
                               (static_cast<std::uint64_t>(words[offset + 1U]) << 32U);
    return static_cast<std::int64_t>(bits);
}

} // namespace

Result<void> run_device_self_test(DeviceView device) {
    if (!device.complete()) {
        return Result<void>::failure(
            {ErrorCode::backend_error, "self-test received an incomplete Vulkan device"});
    }

    constexpr std::int64_t one = numeric::Q32_32::one_raw;
    std::array<ArithmeticCase, case_count> cases{};
    std::size_t case_index = 0;
    const auto append_case = [&](std::uint32_t operation, std::int64_t lhs, std::int64_t rhs) {
        ArithmeticCase arithmetic_case{operation, lhs, rhs, 0, true};
        const numeric::Q32_32 lhs_q32 = numeric::Q32_32::from_raw(lhs);
        const numeric::Q32_32 rhs_q32 = numeric::Q32_32::from_raw(rhs);
        switch (operation) {
        case 0:
            arithmetic_case.expected = numeric::Q32_32::add_saturating(lhs_q32, rhs_q32).raw();
            break;
        case 1:
            arithmetic_case.expected = numeric::Q32_32::subtract_saturating(lhs_q32, rhs_q32).raw();
            break;
        case 2:
            arithmetic_case.expected = numeric::Q32_32::multiply(lhs_q32, rhs_q32).raw();
            break;
        case 3: {
            const auto result = numeric::Q32_32::divide(lhs_q32, rhs_q32);
            arithmetic_case.success = result.has_value();
            if (result) arithmetic_case.expected = result.value().raw();
            break;
        }
        case 4: {
            const auto result = numeric::Q32_32::sqrt(lhs_q32);
            arithmetic_case.success = result.has_value();
            if (result) arithmetic_case.expected = result.value().raw();
            break;
        }
        case 5: {
            const auto result = numeric::Q32_32::log2(lhs_q32);
            arithmetic_case.success = result.has_value();
            if (result) arithmetic_case.expected = result.value().raw();
            break;
        }
        case 6:
            arithmetic_case.expected = numeric::Q32_32::exp2(lhs_q32).raw();
            break;
        case 7: {
            const auto result = numeric::Q32_32::cbrt(lhs_q32);
            arithmetic_case.success = result.has_value();
            if (result) arithmetic_case.expected = result.value().raw();
            break;
        }
        case 8:
            arithmetic_case.expected = numeric::Q32_32::average(lhs_q32, rhs_q32).raw();
            break;
        case 11: {
            const Q48DivideResult result = divide_q48_rne(lhs, rhs);
            arithmetic_case.expected = result.raw;
            arithmetic_case.success = result.success;
            break;
        }
        case 12: {
            const std::uint32_t denominator = static_cast<std::uint32_t>(rhs);
            arithmetic_case.prepared_denominator = denominator;
            arithmetic_case.success = denominator != 0U;
            if (arithmetic_case.success) {
                const std::uint64_t numerator = std::bit_cast<std::uint64_t>(lhs);
                arithmetic_case.expected = std::bit_cast<std::int64_t>(
                    numeric::detail::divide_u64_rne(numerator, denominator));
            }
            break;
        }
        default:
            arithmetic_case.success = false;
            break;
        }
        cases[case_index++] = arithmetic_case;
    };
    const auto append_expected_case = [&cases, &case_index](std::uint32_t operation,
                                                            std::int64_t lhs, std::int64_t rhs,
                                                            std::int64_t expected) {
        cases[case_index++] = ArithmeticCase{operation, lhs, rhs, expected, true, 0U};
    };

    append_case(0, std::numeric_limits<std::int64_t>::max(), 1);
    append_case(0, std::numeric_limits<std::int64_t>::min(), -1);
    append_case(0, one + (one / 2), -(one / 2));
    append_case(1, std::numeric_limits<std::int64_t>::max(), -1);
    append_case(1, std::numeric_limits<std::int64_t>::min(), 1);
    append_case(1, -1, std::numeric_limits<std::int64_t>::min());
    append_case(2, one + (one / 2), one * 2);
    append_case(2, -one - (one / 2), one * 2);
    append_case(2, 1, std::int64_t{1} << 31U);
    append_case(2, 3, std::int64_t{1} << 31U);
    append_case(2, std::numeric_limits<std::int64_t>::max(), one * 2);
    append_case(3, one * 3, one * 2);
    append_case(3, -one * 3, one * 2);
    append_case(3, one, 0);
    append_case(4, one * 4, 0);
    append_case(4, one * 2, 0);
    append_case(4, -one, 0);
    append_case(5, one * 8, 0);
    append_case(5, one / 2, 0);
    append_case(5, 0, 0);
    append_case(6, one / 2, 0);
    append_case(6, -one, 0);
    append_case(7, one * 8, 0);
    append_case(7, -one * 8, 0);
    append_case(8, std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max());
    append_case(8, std::numeric_limits<std::int64_t>::min(),
                std::numeric_limits<std::int64_t>::min());
    append_case(8, std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::min());
    append_case(8, -1, -2);

    constexpr std::int64_t q16_48_one = std::int64_t{1} << 48U;
    append_expected_case(9, -80, 10'386'158, 10'386'158);
    append_expected_case(9, 3 * q16_48_one, 4 * q16_48_one, 5 * q16_48_one);
    append_expected_case(9, std::numeric_limits<std::int64_t>::min(), 0,
                         std::numeric_limits<std::int64_t>::max());
    append_expected_case(10, 3, 0, 2);
    append_expected_case(10, 1, 0, 0);
    append_expected_case(10, 0, 3, -2);
    append_expected_case(10, std::numeric_limits<std::int64_t>::max(),
                         std::numeric_limits<std::int64_t>::min(),
                         std::numeric_limits<std::int64_t>::max());
    append_expected_case(10, std::numeric_limits<std::int64_t>::min(),
                         std::numeric_limits<std::int64_t>::max(),
                         std::numeric_limits<std::int64_t>::min());

    append_case(11, 3 * q16_48_one, 2 * q16_48_one);
    append_case(11, -3 * q16_48_one, 2 * q16_48_one);
    append_case(11, 3 * q16_48_one, -2 * q16_48_one);
    append_case(11, -3 * q16_48_one, -2 * q16_48_one);
    append_case(11, 0, q16_48_one);
    append_case(11, q16_48_one, 0);
    append_case(11, 12'345, 0x8001);
    append_case(11, 0x1234'5678, 0x2345'6789);
    append_case(11, 0x1234'5678'9abcLL, 0x2345'6789'abcdLL);
    append_case(11, 0x1234'5678'9abc'defLL, 0x2345'6789'abcd'ef01LL);
    append_case(11, 0x7fff'ffff'ffff'ffffLL, q16_48_one);
    append_case(11, std::numeric_limits<std::int64_t>::min(), q16_48_one);
    append_case(11, q16_48_one, 1);
    append_case(11, 1, std::numeric_limits<std::int64_t>::max());
    append_case(11, 0x0000'0001'0000'0001LL, 0x0000'0000'ffff'ffffLL);
    append_case(11, 0x0000'ffff'0000'ffffLL, 0x0000'8000'0000'0001LL);
    append_case(11, 0x1111'2222'3333'4444LL, 0x1111'0000'ffff'8001LL);
    append_case(11, 0x4000'0000'0000'0000LL, 0x7fff'ffff'ffff'ffffLL);

    std::uint64_t random_state = 0x4d595df4d0f33173ULL;
    const auto next_random = [&]() {
        random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
        return random_state;
    };
    for (std::size_t random_q48_case = 0U; random_q48_case < 24U; ++random_q48_case) {
        const std::int64_t lhs = static_cast<std::int64_t>(next_random());
        std::int64_t rhs = static_cast<std::int64_t>(next_random());
        if (rhs == 0) rhs = 1;
        append_case(11, lhs, rhs);
    }

    const auto append_prepared_divider_case = [&append_case](std::uint64_t numerator,
                                                             std::uint32_t denominator) {
        append_case(12U, std::bit_cast<std::int64_t>(numerator),
                    static_cast<std::int64_t>(denominator));
    };
    constexpr std::uint64_t maximum_u64 = std::numeric_limits<std::uint64_t>::max();
    append_prepared_divider_case(0U, 1U);
    append_prepared_divider_case(maximum_u64, 1U);
    append_prepared_divider_case(0U, 2U);
    append_prepared_divider_case(1U, 2U);
    append_prepared_divider_case(3U, 2U);
    append_prepared_divider_case(5U, 2U);
    append_prepared_divider_case(maximum_u64, 2U);
    append_prepared_divider_case(3U, 3U);
    append_prepared_divider_case(maximum_u64, 3U);
    constexpr std::uint64_t quotient_sample = 0x1234'5678ULL;
    append_prepared_divider_case(quotient_sample * 7U - 1U, 7U);
    append_prepared_divider_case(quotient_sample * 7U, 7U);
    append_prepared_divider_case(quotient_sample * 7U + 1U, 7U);

    const auto append_prepared_u128_case = [&cases, &case_index](std::uint64_t numerator_low,
                                                                 std::uint64_t numerator_high,
                                                                 std::uint32_t denominator) {
        cases[case_index++] = ArithmeticCase{14U,
                                             std::bit_cast<std::int64_t>(numerator_low),
                                             std::bit_cast<std::int64_t>(numerator_high),
                                             0,
                                             true,
                                             denominator};
    };
    append_prepared_u128_case(0U, 0U, 1U);
    append_prepared_u128_case(maximum_u64, maximum_u64, 1U);
    append_prepared_u128_case(maximum_u64, 0U, 2U);
    append_prepared_u128_case(maximum_u64, maximum_u64, 3U);
    append_prepared_u128_case(0xfedc'ba98'7654'3210ULL, 0x0123'4567'89ab'cdefULL, 0xffff'ffffU);
    append_prepared_u128_case(0x0000'0000'0000'0000ULL, 0x0000'0001'0000'0000ULL, 4'294'967'291U);

    while (case_index < cases.size()) {
        const std::uint32_t operation = static_cast<std::uint32_t>(case_index % 9U);
        std::int64_t lhs =
            static_cast<std::int64_t>(next_random() %
                                      (std::uint64_t{16} * static_cast<std::uint64_t>(one))) -
            (one * 8);
        std::int64_t rhs =
            static_cast<std::int64_t>(next_random() %
                                      (std::uint64_t{16} * static_cast<std::uint64_t>(one))) -
            (one * 8);
        if (operation == 3U && rhs == 0) rhs = one / 3;
        if (operation == 4U) {
            lhs = static_cast<std::int64_t>(next_random() %
                                            (std::uint64_t{100} * static_cast<std::uint64_t>(one)));
        }
        if (operation == 5U) {
            lhs = static_cast<std::int64_t>(
                      next_random() % (std::uint64_t{100} * static_cast<std::uint64_t>(one))) +
                  1;
        }
        if (operation == 6U) {
            lhs = static_cast<std::int64_t>(next_random() %
                                            (std::uint64_t{12} * static_cast<std::uint64_t>(one))) -
                  (one * 6);
        }
        if (operation == 7U) {
            lhs = static_cast<std::int64_t>(
                      next_random() % (std::uint64_t{200} * static_cast<std::uint64_t>(one))) -
                  (one * 100);
        }
        append_case(operation, lhs, rhs);
    }

    std::array<std::uint32_t, case_count * words_per_case> words{};
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const std::size_t base = index * words_per_case;
        words[base] = cases[index].operation;
        words[base + 1U] = 0xdeadbeefu;
        store_raw(words, base + 2U, cases[index].lhs);
        store_raw(words, base + 4U, cases[index].rhs);
        words[base + 6U] = 0xdeadbeefu;
        words[base + 7U] = 0xc001d00du;
        words[base + 8U] = 0xfeedfaceu;
        words[base + 9U] = 0x12345678u;
        if (cases[index].operation == 12U || cases[index].operation == 14U) {
            const numeric::PreparedU32Divider divider =
                numeric::prepare_u32_divider(cases[index].prepared_denominator);
            words[base + 6U] = divider.denominator;
            words[base + 7U] = divider.reciprocal_low;
            words[base + 8U] = divider.reciprocal_high;
            words[base + 9U] = divider.power_of_two_shift;
        }
        words[base + 10U] = 0xdeadbeefu;
        words[base + 11U] = 0xc001d00du;
    }

    BufferCreateInfo buffer_info = host_visible_storage(sizeof(words));
    auto buffer = Buffer::create(device, buffer_info);
    if (!buffer) return Result<void>::failure(buffer.error());
    auto transfer = buffer.value()->write(std::as_bytes(std::span(words)));
    if (!transfer) return transfer;

    auto pipeline = ComputePipeline::create(device, generated::self_test_spv, 1, 0);
    if (!pipeline) return Result<void>::failure(pipeline.error());
    const std::array bindings{BufferBinding{buffer.value().get()}};
    auto updated = pipeline.value()->update_buffers(0, bindings);
    if (!updated) return updated;

    auto commands = CommandContext::create(device);
    if (!commands) return Result<void>::failure(commands.error());
    auto command_buffer = commands.value()->begin();
    if (!command_buffer) return Result<void>::failure(command_buffer.error());
    commands.value()->memory_barrier(VK_PIPELINE_STAGE_HOST_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT,
                                     VK_ACCESS_SHADER_READ_BIT);
    const auto recorded = pipeline.value()->record(command_buffer.value(), 0,
                                                   std::span<const std::byte>{}, {2, 1, 1});
    if (!recorded) return recorded;
    commands.value()->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_ACCESS_HOST_READ_BIT);
    const auto submitted = commands.value()->submit_and_wait();
    if (!submitted) return submitted;
    transfer = buffer.value()->read(std::as_writable_bytes(std::span(words)));
    if (!transfer) return transfer;

    for (std::size_t index = 0; index < cases.size(); ++index) {
        const std::size_t base = index * words_per_case;
        const bool actual_success = words[base + 1U] != 0U;
        const std::int64_t actual = load_raw(words, base + 10U);
        if (actual_success != cases[index].success ||
            (actual_success && actual != cases[index].expected)) {
            std::ostringstream stream;
            stream << "fixed-point operation " << cases[index].operation << " case " << index
                   << " expected status " << cases[index].success << " raw "
                   << cases[index].expected << " but GPU returned status " << actual_success
                   << " raw " << actual;
            return Result<void>::failure({ErrorCode::device_self_test_failed, stream.str()});
        }
    }
    return Result<void>::success();
}

} // namespace crossvmaf::gpu
