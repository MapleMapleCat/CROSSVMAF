if(CROSSVMAF_BUILD_TOOLS)
add_library(crossvmaf_reference_verification STATIC
    ${PROJECT_SOURCE_DIR}/apps/conformance/reference_evaluator.cpp
)
crossvmaf_use_project_headers(crossvmaf_reference_verification)
crossvmaf_configure_target(crossvmaf_reference_verification)
target_link_libraries(crossvmaf_reference_verification PUBLIC crossvmaf_scoring crossvmaf_adm3_reference crossvmaf_cambi_reference crossvmaf_motion3_reference crossvmaf_speed_chroma_reference)
target_include_directories(crossvmaf_reference_verification PUBLIC ${PROJECT_SOURCE_DIR}/apps/conformance)
endif()

if(CROSSVMAF_BUILD_TOOLS)
    add_subdirectory(crossvmaf)
    add_subdirectory(conformance)
endif()

if(CROSSVMAF_BUILD_STUDIO)
    add_subdirectory(studio)
endif()
