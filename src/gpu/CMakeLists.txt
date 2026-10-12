add_library(crossvmaf_gpu STATIC
    buffer.cpp
    command.cpp
    pipeline.cpp
    device.cpp
    memory.cpp
    self_test.cpp
    submission.cpp
    timestamp.cpp
)
crossvmaf_use_project_headers(crossvmaf_gpu)
crossvmaf_configure_target(crossvmaf_gpu)
target_link_libraries(crossvmaf_gpu PUBLIC volk::volk PRIVATE crossvmaf_numeric)
target_compile_definitions(crossvmaf_gpu PUBLIC VK_NO_PROTOTYPES)
crossvmaf_compile_shader(crossvmaf_gpu src/gpu/shaders/self_test.comp self_test_spv)

add_custom_target(crossvmaf_gpu_layer_check
    COMMAND ${CMAKE_COMMAND}
        "-DRUNTIME_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}"
        -P "${PROJECT_SOURCE_DIR}/cmake/vulkan/CheckRuntimeLayer.cmake"
    VERBATIM)
add_dependencies(crossvmaf_gpu crossvmaf_gpu_layer_check)

target_compile_definitions(crossvmaf_gpu PRIVATE ${crossvmaf_decode_definitions})
