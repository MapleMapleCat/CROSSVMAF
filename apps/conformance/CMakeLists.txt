add_executable(crossvmaf_conformance
    main.cpp
    synthetic_frame.cpp
    speed_chroma_probe.cpp
    sha256.cpp
    ${PROJECT_SOURCE_DIR}/apps/conformance/frame_uploader.cpp
)
set_target_properties(crossvmaf_conformance PROPERTIES OUTPUT_NAME crossvmaf-conformance)
crossvmaf_configure_target(crossvmaf_conformance)
target_link_libraries(crossvmaf_conformance PRIVATE
    CrossVMAF::crossvmaf
    # Deliberate white-box dependencies: the conformance probe compares the
    # model definition, CPU verifier, and Vulkan implementation below the API.
    crossvmaf_numeric
    crossvmaf_scoring
    crossvmaf_reference_verification
    crossvmaf
    volk::volk
)
target_compile_definitions(crossvmaf_conformance PRIVATE VK_NO_PROTOTYPES)
target_include_directories(crossvmaf_conformance PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)
crossvmaf_stage_application_licenses(crossvmaf_conformance)
