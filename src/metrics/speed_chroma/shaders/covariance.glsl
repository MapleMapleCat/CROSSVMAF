// Highpass production quantizes int32 * 2^12, hence |sample| <= 2^43.
// With N <= 2^19, the exact wide-integer mean accumulation cannot round or
// saturate: |mean| < 2^44, including the final division/quantization.
// Neither the subtraction nor a sample sum can saturate:
// |sample - mean| < 2^45 and |sum(sample)| <= 2^62.
// Each corrected product-sum term is at most 2^107; their absolute sum
// is below 2^109, safely inside signed 128 bits. No rounding is moved.

void initialize_covariance_displacement(uint displacement_index) {
    if (displacement_index >= 41u) return;
    const uint local_lane = gl_LocalInvocationID.x;
    const uint row_displacement = displacement_index < 5u
        ? 0u : 1u + (displacement_index - 5u) / 9u;
    const int column_displacement = displacement_index < 5u
        ? int(displacement_index) : int((displacement_index - 5u) % 9u) - 4;
    const uint first_column = uint(max(-column_displacement, 0));
    const uint column_count = block_size - uint(abs(column_displacement));
    const uint row_count = block_size - row_displacement;
    const uint first_row = local_lane / column_count;
    const uint column_position = local_lane % column_count;
    const bool owns_entry = first_row < row_count;
    const uint source_row_count = parameters.submatrix_height + row_count - 1u;
    const uint row_distance = row_displacement * parameters.operating_width;

    uvec4 product_sum = uvec4(0u);
    uvec2 first_sample_sum = uvec2(0u);
    uvec2 second_sample_sum = uvec2(0u);
    for (uint batch_row = 0u; batch_row < source_row_count; batch_row += 64u) {
        const uint source_row = batch_row + local_lane;
        if (source_row < source_row_count) {
            const uint first_source = source_row * parameters.operating_width + first_column;
            const uint second_source = uint(int(first_source + row_distance) + column_displacement);
            uvec4 row_product_sum = uvec4(0u);
            uvec2 first_row_sum = uvec2(0u);
            uvec2 second_row_sum = uvec2(0u);
            for (uint column = 0u; column < parameters.submatrix_width; ++column) {
                const uvec2 first_sample = load_quantized_feature_sample(first_source + column);
                const uvec2 second_sample = load_quantized_feature_sample(second_source + column);
                row_product_sum = cv_u128_add(row_product_sum,
                    cv_i64_multiply_wide(first_sample, second_sample));
                first_row_sum = cv_u64_add(first_row_sum, first_sample);
                second_row_sum = cv_u64_add(second_row_sum, second_sample);
            }
            for (uint position = 0u; position < column_count; ++position) {
                const uint shared_index = position * 64u + local_lane;
                covariance_partials[shared_index] = row_product_sum;
                covariance_row_sample_sums[shared_index] = uvec4(first_row_sum, second_row_sum);
                if (position + 1u < column_count) {
                    const uvec2 outgoing_first = load_quantized_feature_sample(first_source + position);
                    const uvec2 outgoing_second = load_quantized_feature_sample(second_source + position);
                    const uvec2 incoming_first = load_quantized_feature_sample(
                        first_source + position + parameters.submatrix_width);
                    const uvec2 incoming_second = load_quantized_feature_sample(
                        second_source + position + parameters.submatrix_width);
                    row_product_sum = cv_u128_add(cv_u128_subtract(row_product_sum,
                        cv_i64_multiply_wide(outgoing_first, outgoing_second)),
                        cv_i64_multiply_wide(incoming_first, incoming_second));
                    first_row_sum = cv_u64_add(cv_u64_subtract(first_row_sum, outgoing_first), incoming_first);
                    second_row_sum = cv_u64_add(cv_u64_subtract(second_row_sum, outgoing_second), incoming_second);
                }
            }
        }
        barrier();
        if (owns_entry) {
            const uint begin_row = max(batch_row, first_row);
            const uint end_row = min(batch_row + 64u, first_row + parameters.submatrix_height);
            for (uint row = begin_row; row < end_row; ++row) {
                const uint shared_index = column_position * 64u + row - batch_row;
                product_sum = cv_u128_add(product_sum, covariance_partials[shared_index]);
                const uvec4 sample_sums = covariance_row_sample_sums[shared_index];
                first_sample_sum = cv_u64_add(first_sample_sum, sample_sums.xy);
                second_sample_sum = cv_u64_add(second_sample_sum, sample_sums.zw);
            }
        }
        barrier();
    }
    if (!owns_entry) return;
    const uint first_index = first_row * block_size + first_column + column_position;
    const uint second_index = uint(int(first_index + row_displacement * block_size) + column_displacement);
    const uvec2 first_mean = scratch.values[current_scratch_index(means_logical_offset + first_index)];
    const uvec2 second_mean = scratch.values[current_scratch_index(means_logical_offset + second_index)];
    const uvec2 scaled_first_mean = cv_i64_multiply_wide(
        first_mean, uvec2(parameters.covariance_sample_count, 0u)).xy;
    product_sum = cv_u128_subtract(product_sum, cv_i64_multiply_wide(second_mean, first_sample_sum));
    product_sum = cv_u128_subtract(product_sum, cv_i64_multiply_wide(first_mean, second_sample_sum));
    product_sum = cv_u128_add(product_sum, cv_i64_multiply_wide(scaled_first_mean, second_mean));
    store_covariance_entry(first_index, second_index, true, product_sum);
}
