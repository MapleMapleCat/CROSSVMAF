add_library(crossvmaf_frame STATIC
    frame.cpp
)
crossvmaf_use_project_headers(crossvmaf_frame)
crossvmaf_configure_target(crossvmaf_frame)
target_link_libraries(crossvmaf_frame PUBLIC Vulkan::Headers)

add_library(crossvmaf_video STATIC
    frame.cpp
    ffmpeg.cpp
    ffmpeg_abi.cpp
    decoder.cpp
    decode_backend.cpp
    backend_selection.cpp
    decode_device.cpp
    frame_transfer.cpp
    transfers/d3d11_vulkan.cpp
    backends/nvdec.cpp
    backends/qsv.cpp
    backends/amf.cpp
    backends/vaapi.cpp
    backends/software.cpp
    probe.cpp
    backends/vulkan.cpp
    backends/vulkan_dispatch.cpp
    vulkan_convert.cpp
)
crossvmaf_use_project_headers(crossvmaf_video)
crossvmaf_configure_target(crossvmaf_video)
target_link_libraries(crossvmaf_video PUBLIC crossvmaf_frame crossvmaf_gpu)
target_link_libraries(crossvmaf_video PRIVATE FFmpeg::avformat FFmpeg::avcodec FFmpeg::avutil Threads::Threads)
target_compile_definitions(crossvmaf_video PRIVATE VK_NO_PROTOTYPES CROSSVMAF_FRAME_NATIVE)
target_compile_definitions(crossvmaf_video PRIVATE ${crossvmaf_decode_definitions})
if(WIN32)
    target_link_libraries(crossvmaf_video PRIVATE dxgi)
endif()
if(CROSSVMAF_ENABLE_NVDEC)
    target_include_directories(crossvmaf_video SYSTEM PRIVATE "${crossvmaf_nvcodec_SOURCE_DIR}/include")
    target_link_libraries(crossvmaf_video PRIVATE ${CMAKE_DL_LIBS})
endif()
if(CROSSVMAF_ENABLE_AMF)
    target_include_directories(crossvmaf_video SYSTEM PRIVATE "${crossvmaf_amf_include}")
endif()
crossvmaf_compile_shader(
    crossvmaf_video
    src/video/shaders/semiplanar_to_planar.comp
    semiplanar_nv12_spv
    OUTPUT_NAME semiplanar_nv12
    DEFINES CROSSVMAF_P010=0
    EXPECTED_LOCAL_SIZE_X 64)
crossvmaf_compile_shader(
    crossvmaf_video
    src/video/shaders/semiplanar_to_planar.comp
    semiplanar_p010_spv
    OUTPUT_NAME semiplanar_p010
    DEFINES CROSSVMAF_P010=1
    EXPECTED_LOCAL_SIZE_X 64)
