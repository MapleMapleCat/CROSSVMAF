set(CROSSVMAF_CLI_SOURCES
    main.cpp
    measurement.cpp
    cpu_verification.cpp
    hotspot_profiler.cpp
    vulkan_hotspot_report.cpp
    vulkan_hotspot_analysis.cpp
    vulkan_hotspot_reporter.cpp
    options.cpp
    output.cpp
)

add_executable(crossvmaf_cli ${CROSSVMAF_CLI_SOURCES})
set_target_properties(crossvmaf_cli PROPERTIES OUTPUT_NAME crossvmaf)
crossvmaf_configure_target(crossvmaf_cli)
target_link_libraries(crossvmaf_cli PRIVATE
    CrossVMAF::crossvmaf
    CrossVMAF::version
    crossvmaf_reference_verification
    Threads::Threads
    Vulkan::Headers
    nlohmann_json::nlohmann_json
)
target_include_directories(crossvmaf_cli PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)
crossvmaf_stage_application_licenses(crossvmaf_cli)
