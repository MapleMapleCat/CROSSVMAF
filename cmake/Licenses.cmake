include_guard(GLOBAL)

function(crossvmaf_stage_license target source_file output_name)
    if(NOT EXISTS "${source_file}")
        message(FATAL_ERROR "Missing distribution license: ${source_file}")
    endif()
    set_property(TARGET ${target} APPEND PROPERTY CROSSVMAF_LICENSE_FILES
        "${source_file}\tlicenses/${output_name}")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory
            "$<TARGET_FILE_DIR:${target}>/licenses"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${source_file}" "$<TARGET_FILE_DIR:${target}>/licenses/${output_name}"
        VERBATIM
    )
endfunction()

function(crossvmaf_stage_application_licenses target)
    if(CROSSVMAF_ENABLE_AMF)
        crossvmaf_stage_license(${target} "${crossvmaf_amf_SOURCE_DIR}/LICENSE.txt" "AMD-AMF-MIT.txt")
    endif()
    if(CROSSVMAF_ENABLE_QSV)
        crossvmaf_stage_license(${target} "${crossvmaf_vpl_SOURCE_DIR}/LICENSE" "Intel-VPL-MIT.txt")
        crossvmaf_stage_license(${target} "${crossvmaf_vpl_SOURCE_DIR}/third-party-programs.txt" "Intel-VPL-third-party-programs.txt")
    endif()
    if(CROSSVMAF_ENABLE_NVDEC)
        crossvmaf_stage_license(${target} "${crossvmaf_nvcodec_SOURCE_DIR}/include/ffnvcodec/dynlink_loader.h" "nv-codec-headers-NOTICE.h")
    endif()
    crossvmaf_stage_codec_licenses(${target})
    crossvmaf_stage_license(${target}
        "${PROJECT_SOURCE_DIR}/LICENSE" "CrossVMAF-BSD-2-Clause-Patent.txt")
    crossvmaf_stage_license(${target}
        "${PROJECT_SOURCE_DIR}/third_party/licenses/Netflix-VMAF-BSD-2-Clause-Patent.txt"
        "Netflix-VMAF-BSD-2-Clause-Patent.txt")
    crossvmaf_stage_license(${target}
        "${volk_SOURCE_DIR}/LICENSE.md" "volk-MIT.txt")
    crossvmaf_stage_license(${target}
        "${vulkanheaders_SOURCE_DIR}/LICENSE.md" "Vulkan-Headers-NOTICE.md")
    crossvmaf_stage_license(${target}
        "${vulkanheaders_SOURCE_DIR}/LICENSES/MIT.txt" "Vulkan-Headers-MIT.txt")
    crossvmaf_stage_license(${target}
        "${vulkanheaders_SOURCE_DIR}/LICENSES/Apache-2.0.txt" "Vulkan-Headers-Apache-2.0.txt")
    crossvmaf_stage_license(${target}
        "${nlohmann_json_SOURCE_DIR}/LICENSE.MIT" "nlohmann-json-MIT.txt")
endfunction()

function(crossvmaf_stage_studio_licenses target)
    crossvmaf_stage_application_licenses(${target})
    # Slint is populated inside a variable-scoped block in Dependencies.cmake.
    FetchContent_GetProperties(slint SOURCE_DIR slint_source_directory)
    crossvmaf_stage_license(${target}
        "${PROJECT_SOURCE_DIR}/apps/studio/ui/fonts/OFL-NotoSansSC.txt" "OFL-NotoSansSC.txt")
    crossvmaf_stage_license(${target}
        "${PROJECT_SOURCE_DIR}/apps/studio/ui/fonts/OFL-Inter.txt" "OFL-Inter.txt")
    crossvmaf_stage_license(${target}
        "${PROJECT_SOURCE_DIR}/apps/studio/ui/fonts/OFL-JetBrainsMono.txt" "OFL-JetBrainsMono.txt")
    crossvmaf_stage_license(${target}
        "${PROJECT_SOURCE_DIR}/apps/studio/ui/icons/LICENSE-lucide.txt" "LICENSE-lucide.txt")
    crossvmaf_stage_license(${target}
        "${slint_source_directory}/LICENSE.md" "Slint-LICENSE.md")
    crossvmaf_stage_license(${target}
        "${slint_source_directory}/LICENSES/LicenseRef-Slint-Royalty-free-2.0.md"
        "Slint-Royalty-free-2.0.md")
endfunction()
