add_library(crossvmaf_speed_chroma_reference STATIC
    linear_algebra.cpp
    reference.cpp
    geometry.cpp
    statistics.cpp
)
crossvmaf_use_project_headers(crossvmaf_speed_chroma_reference)
crossvmaf_configure_target(crossvmaf_speed_chroma_reference)
target_link_libraries(crossvmaf_speed_chroma_reference PUBLIC crossvmaf_numeric crossvmaf_frame)

add_library(crossvmaf_speed_chroma_vulkan STATIC
    vulkan.cpp
)
crossvmaf_use_project_headers(crossvmaf_speed_chroma_vulkan)
crossvmaf_configure_target(crossvmaf_speed_chroma_vulkan)
target_link_libraries(crossvmaf_speed_chroma_vulkan PUBLIC crossvmaf_gpu crossvmaf_numeric crossvmaf_frame volk::volk)
target_compile_definitions(crossvmaf_speed_chroma_vulkan PRIVATE VK_NO_PROTOTYPES)

crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/scale.comp
    speed_chroma_scale_spv
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sampling.glsl"
    MAX_UDIV_COUNT 0
    OUTPUT_NAME speed_chroma_scale)
crossvmaf_compile_shader(crossvmaf_speed_chroma_vulkan src/metrics/speed_chroma/shaders/vertical_filter.comp speed_chroma_filter_vertical_spv
    OUTPUT_NAME speed_chroma_filter_vertical)
crossvmaf_compile_shader(crossvmaf_speed_chroma_vulkan src/metrics/speed_chroma/shaders/antialias.comp speed_chroma_antialias_fused_spv
    OUTPUT_NAME speed_chroma_antialias_fused)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/antialias.comp
    speed_chroma_antialias_identity_spv
    OUTPUT_NAME speed_chroma_antialias_identity
    DEFINES CROSSVMAF_SPEED_CHROMA_IDENTITY_SCALE=1
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sampling.glsl"
    MAX_UDIV_COUNT 0)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/antialias.comp
    speed_chroma_antialias_identity_products_spv
    OUTPUT_NAME speed_chroma_antialias_identity_products
    DEFINES
        CROSSVMAF_SPEED_CHROMA_IDENTITY_SCALE=1
        CROSSVMAF_SPEED_CHROMA_IDENTITY_PRODUCTS=1
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sampling.glsl"
    MAX_UDIV_COUNT 0)
crossvmaf_compile_shader(crossvmaf_speed_chroma_vulkan src/metrics/speed_chroma/shaders/local_filter.comp speed_chroma_local_horizontal_spv
    OUTPUT_NAME speed_chroma_local_horizontal)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_means_spv
    OUTPUT_NAME speed_chroma_means
    DEFINES
        CROSSVMAF_SPEED_CHROMA_PIPELINE=1
        CROSSVMAF_SPEED_CHROMA_MEANS_LOCAL_SIZE=64
    EXPECTED_LOCAL_SIZE_X 64
    AUDIT_MEANS_WIDE_EXACT
    AUDIT_MEANS_ROW_REDUCTION)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_compact_tridiagonal_spv
    OUTPUT_NAME speed_chroma_compact_tridiagonal
    DEFINES
        CROSSVMAF_SPEED_CHROMA_PIPELINE=2
        CROSSVMAF_SPEED_CHROMA_COMPACT_TRIDIAGONAL_LOCAL_SIZE=64
    EXPECTED_LOCAL_SIZE_X 64
    MAX_UDIV_COUNT 3
    AUDIT_COMPACT_COOPERATIVE)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_compact_tridiagonal_wave_spv
    OUTPUT_NAME speed_chroma_compact_tridiagonal_wave
    DEFINES
        CROSSVMAF_SPEED_CHROMA_PIPELINE=2
        CROSSVMAF_SPEED_CHROMA_COMPACT_TRIDIAGONAL_LOCAL_SIZE=64
        CROSSVMAF_SPEED_CHROMA_STREAMING_WAVE=1
    EXPECTED_LOCAL_SIZE_X 64
    MAX_UDIV_COUNT 3
    AUDIT_COMPACT_COOPERATIVE)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_exact_regularity_spv
    OUTPUT_NAME speed_chroma_exact_regularity
    DEFINES CROSSVMAF_SPEED_CHROMA_PIPELINE=6
    EXPECTED_LOCAL_SIZE_X 32
    AUDIT_EXACT_WIDE_SHARED)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_exact_regularity_wave_spv
    OUTPUT_NAME speed_chroma_exact_regularity_wave
    DEFINES
        CROSSVMAF_SPEED_CHROMA_PIPELINE=6
        CROSSVMAF_SPEED_CHROMA_STREAMING_WAVE=1
    EXPECTED_LOCAL_SIZE_X 32
    AUDIT_EXACT_WIDE_SHARED)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_compact_blocks_spv
    OUTPUT_NAME speed_chroma_compact_blocks
    DEFINES
        CROSSVMAF_SPEED_CHROMA_PIPELINE=3
        CROSSVMAF_SPEED_CHROMA_BLOCK_STATISTICS_LOCAL_SIZE=32
    EXPECTED_LOCAL_SIZE_X 32
    MAX_UDIV_COUNT 4
    AUDIT_BLOCK_TRANSFORMED_SHARED)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_block_statistics_scalar_spv
    OUTPUT_NAME speed_chroma_block_statistics_scalar
    DEFINES CROSSVMAF_SPEED_CHROMA_PIPELINE=7
    EXPECTED_LOCAL_SIZE_X 64
    MAX_UDIV_COUNT 4
    AUDIT_BLOCK_SCALAR)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_block_statistics_scalar_wave_spv
    OUTPUT_NAME speed_chroma_block_statistics_scalar_wave
    DEFINES
        CROSSVMAF_SPEED_CHROMA_PIPELINE=7
        CROSSVMAF_SPEED_CHROMA_STREAMING_WAVE=1
    EXPECTED_LOCAL_SIZE_X 64
    MAX_UDIV_COUNT 4
    AUDIT_BLOCK_SCALAR)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_compact_blocks_wave_spv
    OUTPUT_NAME speed_chroma_compact_blocks_wave
    DEFINES
        CROSSVMAF_SPEED_CHROMA_PIPELINE=3
        CROSSVMAF_SPEED_CHROMA_BLOCK_STATISTICS_LOCAL_SIZE=32
        CROSSVMAF_SPEED_CHROMA_STREAMING_WAVE=1
    EXPECTED_LOCAL_SIZE_X 32
    MAX_UDIV_COUNT 4
    AUDIT_BLOCK_TRANSFORMED_SHARED)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_finalize_spv
    OUTPUT_NAME speed_chroma_finalize
    DEFINES CROSSVMAF_SPEED_CHROMA_PIPELINE=4
    EXPECTED_LOCAL_SIZE_X 64)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_finalize_wave_spv
    OUTPUT_NAME speed_chroma_finalize_wave
    DEFINES
        CROSSVMAF_SPEED_CHROMA_PIPELINE=4
        CROSSVMAF_SPEED_CHROMA_STREAMING_WAVE=1
    EXPECTED_LOCAL_SIZE_X 64)
crossvmaf_compile_shader(
    crossvmaf_speed_chroma_vulkan
    src/metrics/speed_chroma/shaders/score.comp
    speed_chroma_covariance_spv
    OUTPUT_NAME speed_chroma_covariance
    DEFINES CROSSVMAF_SPEED_CHROMA_PIPELINE=5
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/covariance.glsl"
    EXPECTED_LOCAL_SIZE_X 64)
