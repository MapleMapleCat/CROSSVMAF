add_library(crossvmaf STATIC
    evaluation_stream.cpp
    frame_pair_record.cpp
    frame_records.cpp
    input_lease_pool.cpp
    input_staging_ring.cpp
    session_telemetry.cpp
    validation.cpp
)
crossvmaf_use_project_headers(crossvmaf)
crossvmaf_configure_target(crossvmaf)
target_link_libraries(crossvmaf PUBLIC crossvmaf_scoring crossvmaf_video crossvmaf_gpu crossvmaf_adm3_vulkan crossvmaf_cambi_vulkan crossvmaf_motion3_vulkan crossvmaf_speed_chroma_vulkan Threads::Threads volk::volk)

add_library(CrossVMAF::crossvmaf ALIAS crossvmaf)
target_compile_definitions(crossvmaf PRIVATE VK_NO_PROTOTYPES)
crossvmaf_generate_spirv_manifest(crossvmaf)
target_include_directories(crossvmaf PRIVATE ${PROJECT_BINARY_DIR}/src/video/generated)
