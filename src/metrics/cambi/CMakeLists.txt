add_library(crossvmaf_cambi_reference STATIC
    c_values.cpp
    reference.cpp
)
crossvmaf_use_project_headers(crossvmaf_cambi_reference)
crossvmaf_configure_target(crossvmaf_cambi_reference)
target_link_libraries(crossvmaf_cambi_reference PUBLIC crossvmaf_numeric crossvmaf_frame)

add_library(crossvmaf_cambi_vulkan STATIC
    vulkan.cpp
)
crossvmaf_use_project_headers(crossvmaf_cambi_vulkan)
crossvmaf_configure_target(crossvmaf_cambi_vulkan)
target_link_libraries(crossvmaf_cambi_vulkan PUBLIC crossvmaf_gpu crossvmaf_numeric crossvmaf_frame volk::volk)
target_compile_definitions(crossvmaf_cambi_vulkan PRIVATE VK_NO_PROTOTYPES)

crossvmaf_compile_shader(crossvmaf_cambi_vulkan src/metrics/cambi/shaders/preprocess.comp cambi_preprocess_spv
    OUTPUT_NAME cambi_preprocess)
crossvmaf_compile_shader(crossvmaf_cambi_vulkan src/metrics/cambi/shaders/mask.comp cambi_mask_spv
    OUTPUT_NAME cambi_mask)
crossvmaf_compile_shader(crossvmaf_cambi_vulkan src/metrics/cambi/shaders/decimate.comp cambi_decimate_spv
    OUTPUT_NAME cambi_decimate)
crossvmaf_compile_shader(crossvmaf_cambi_vulkan src/metrics/cambi/shaders/mode.comp cambi_mode_spv
    OUTPUT_NAME cambi_mode)
crossvmaf_compile_shader(
    crossvmaf_cambi_vulkan
    src/metrics/cambi/shaders/c_values.comp
    cambi_c_values_rows_spv
    EXPECTED_LOCAL_SIZE_X 32
    EXPECTED_UDIV_COUNT 2
    AUDIT_CAMBI_CVALUE_STAGED
    OUTPUT_NAME cambi_c_values_rows)
crossvmaf_compile_shader(
    crossvmaf_cambi_vulkan
    src/metrics/cambi/shaders/c_values.comp
    cambi_c_values_rows_u16_spv
    OUTPUT_NAME cambi_c_values_rows_u16
    DEFINES CAMBI_HISTOGRAM_UINT16=1
    EXPECTED_LOCAL_SIZE_X 32
    EXPECTED_UDIV_COUNT 2
    AUDIT_CAMBI_CVALUE_STAGED
    AUDIT_CAMBI_CVALUE_INT16)
crossvmaf_compile_shader(crossvmaf_cambi_vulkan src/metrics/cambi/shaders/radix_init.comp cambi_radix_init_spv
    OUTPUT_NAME cambi_radix_init)
crossvmaf_compile_shader(crossvmaf_cambi_vulkan src/metrics/cambi/shaders/radix_histogram.comp cambi_radix_histogram_spv
    OUTPUT_NAME cambi_radix_histogram)
crossvmaf_compile_shader(crossvmaf_cambi_vulkan src/metrics/cambi/shaders/radix_histogram_2d.comp cambi_radix_histogram_2d_spv
    OUTPUT_NAME cambi_radix_histogram_2d)
crossvmaf_compile_shader(
    crossvmaf_cambi_vulkan
    src/metrics/cambi/shaders/radix_update.comp
    cambi_radix_update_spv
    EXPECTED_LOCAL_SIZE_X 32
    OUTPUT_NAME cambi_radix_update)
crossvmaf_compile_shader(crossvmaf_cambi_vulkan src/metrics/cambi/shaders/pool_reduce.comp cambi_pool_reduce_spv
    OUTPUT_NAME cambi_pool_reduce)
crossvmaf_compile_shader(crossvmaf_cambi_vulkan src/metrics/cambi/shaders/pool_finalize.comp cambi_pool_finalize_spv
    OUTPUT_NAME cambi_pool_finalize)
crossvmaf_compile_shader(crossvmaf_cambi_vulkan src/metrics/cambi/shaders/combine.comp cambi_combine_spv
    OUTPUT_NAME cambi_combine)
