add_library(crossvmaf_adm3_reference STATIC
    reference.cpp
    wavelet.cpp
)
crossvmaf_use_project_headers(crossvmaf_adm3_reference)
crossvmaf_configure_target(crossvmaf_adm3_reference)
target_link_libraries(crossvmaf_adm3_reference PUBLIC crossvmaf_numeric crossvmaf_frame)

add_library(crossvmaf_adm3_vulkan STATIC
    vulkan.cpp
)
crossvmaf_use_project_headers(crossvmaf_adm3_vulkan)
crossvmaf_configure_target(crossvmaf_adm3_vulkan)
target_link_libraries(crossvmaf_adm3_vulkan PUBLIC crossvmaf_gpu crossvmaf_numeric crossvmaf_frame volk::volk)
target_compile_definitions(crossvmaf_adm3_vulkan PRIVATE VK_NO_PROTOTYPES)

crossvmaf_compile_shader(crossvmaf_adm3_vulkan src/metrics/adm3/shaders/wavelet.comp adm3_dwt_spv
    OUTPUT_NAME adm3_dwt)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/decouple.comp
    adm3_decouple_spv
    EXPECTED_UDIV_COUNT 1
    OUTPUT_NAME adm3_decouple)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/segments.comp
    adm3_metric_segments_score_spv
    OUTPUT_NAME adm3_metric_segments_score
    DEFINES CROSSVMAF_ADM3_SCORE_ONLY=1
    EXPECTED_LOCAL_SIZE_X 32)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/segments.comp
    adm3_metric_segments_detailed_spv
    OUTPUT_NAME adm3_metric_segments_detailed
    EXPECTED_LOCAL_SIZE_X 32)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/merge_rows.comp
    adm3_merge_rows_spv
    EXPECTED_LOCAL_SIZE_X 32
    OUTPUT_NAME adm3_merge_rows)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/merge_rows.comp
    adm3_merge_rows_stream_spv
    OUTPUT_NAME adm3_merge_rows_stream
    DEFINES CROSSVMAF_ADM3_STREAMING_PREFIX=1
    EXPECTED_LOCAL_SIZE_X 32)
crossvmaf_compile_shader(crossvmaf_adm3_vulkan src/metrics/adm3/shaders/reduce.comp adm3_reduce_spv
    OUTPUT_NAME adm3_reduce)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/reduce.comp
    adm3_reduce_wave_spv
    OUTPUT_NAME adm3_reduce_wave
    DEFINES CROSSVMAF_ADM3_STREAMING_WAVE=1
    EXPECTED_LOCAL_SIZE_X 256)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/pack_scale.comp
    adm3_pack_scale_score_spv
    OUTPUT_NAME adm3_pack_scale_score
    DEFINES CROSSVMAF_ADM3_SCORE_ONLY=1
    EXPECTED_LOCAL_SIZE_X 16)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/pack_scale.comp
    adm3_pack_scale_detailed_spv
    OUTPUT_NAME adm3_pack_scale_detailed
    EXPECTED_LOCAL_SIZE_X 16)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/pack_scale.comp
    adm3_pack_scale_score_wave_spv
    OUTPUT_NAME adm3_pack_scale_score_wave
    DEFINES
        CROSSVMAF_ADM3_SCORE_ONLY=1
        CROSSVMAF_ADM3_STREAMING_WAVE=1
    EXPECTED_LOCAL_SIZE_X 16)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/pack_scale.comp
    adm3_pack_scale_detailed_wave_spv
    OUTPUT_NAME adm3_pack_scale_detailed_wave
    DEFINES CROSSVMAF_ADM3_STREAMING_WAVE=1
    EXPECTED_LOCAL_SIZE_X 16)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/finalize.comp
    adm3_finalize_batch_spv
    EXPECTED_LOCAL_SIZE_X 16
    AUDIT_ADM3_FINALIZE_COOPERATIVE
    OUTPUT_NAME adm3_finalize_batch)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/finalize.comp
    adm3_finalize_scales_wave_spv
    OUTPUT_NAME adm3_finalize_scales_wave
    DEFINES CROSSVMAF_ADM3_STREAMING_WAVE=1
    EXPECTED_LOCAL_SIZE_X 16
    AUDIT_ADM3_FINALIZE_COOPERATIVE)
crossvmaf_compile_shader(crossvmaf_adm3_vulkan src/metrics/adm3/shaders/combine.comp adm3_combine_spv
    OUTPUT_NAME adm3_combine)
crossvmaf_compile_shader(
    crossvmaf_adm3_vulkan
    src/metrics/adm3/shaders/combine.comp
    adm3_combine_wave_spv
    OUTPUT_NAME adm3_combine_wave
    DEFINES CROSSVMAF_ADM3_STREAMING_WAVE=1
    EXPECTED_LOCAL_SIZE_X 1)
