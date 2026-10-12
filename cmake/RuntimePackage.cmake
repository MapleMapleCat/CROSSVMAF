# Collect canonical targets and original license files, never stale out/ files.
set(crossvmaf_package_targets)
foreach(target crossvmaf_cli crossvmaf_studio)
    if(TARGET ${target})
        list(APPEND crossvmaf_package_targets ${target})
    endif()
endforeach()
if(NOT crossvmaf_package_targets)
    return()
endif()
set(crossvmaf_package_inputs "")
foreach(target IN LISTS crossvmaf_package_targets)
    string(APPEND crossvmaf_package_inputs "$<TARGET_FILE:${target}>\t$<TARGET_FILE_NAME:${target}>\n")
    get_target_property(licenses ${target} CROSSVMAF_LICENSE_FILES)
    foreach(license IN LISTS licenses)
        string(APPEND crossvmaf_package_inputs "${license}\n")
    endforeach()
endforeach()
set(crossvmaf_package_runtime_targets crossvmaf_ffmpeg)
if(CROSSVMAF_ENABLE_QSV)
    list(APPEND crossvmaf_package_runtime_targets CrossVMAF::vpl)
endif()
if(TARGET crossvmaf_studio AND TARGET slint_cpp-shared)
    list(APPEND crossvmaf_package_runtime_targets slint_cpp-shared)
endif()
foreach(target IN LISTS crossvmaf_package_runtime_targets)
    string(APPEND crossvmaf_package_inputs "$<TARGET_FILE:${target}>\t$<TARGET_FILE_NAME:${target}>\n")
endforeach()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    string(APPEND crossvmaf_package_inputs
        "$<TARGET_FILE:crossvmaf_ffmpeg>\t$<TARGET_SONAME_FILE_NAME:crossvmaf_ffmpeg>\n")
endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/package-inputs/$<CONFIG>.txt"
    CONTENT "${crossvmaf_package_inputs}")
add_custom_target(crossvmaf_package
    COMMAND "${Python3_EXECUTABLE}" -B "${PROJECT_SOURCE_DIR}/tools/package_runtime.py"
        --inputs "${CMAKE_BINARY_DIR}/package-inputs/$<CONFIG>.txt"
        --output-root "${CMAKE_BINARY_DIR}/packages/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}/$<CONFIG>"
        --source "${PROJECT_SOURCE_DIR}" --config "$<CONFIG>"
        --compiler "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}"
        --reader "${CROSSVMAF_LLVM_READOBJ}" --contract "${CROSSVMAF_FFMPEG_ABI_CONTRACT}"
        --abi-major "${CROSSVMAF_FFMPEG_ABI_MAJOR}"
        --ffmpeg-name "$<TARGET_FILE_NAME:crossvmaf_ffmpeg>"
        --backend "qsv=${CROSSVMAF_ENABLE_QSV}" --backend "nvdec=${CROSSVMAF_ENABLE_NVDEC}"
        --backend "vulkan_video=${CROSSVMAF_ENABLE_VULKAN_VIDEO}" --backend "vaapi=${CROSSVMAF_ENABLE_VAAPI}"
    DEPENDS ${crossvmaf_package_targets} crossvmaf_ffmpeg_abi_check
    VERBATIM USES_TERMINAL)

# Preserve the documented convenient launch location, while builds/packages
# use the canonical configuration-specific executable above.
if(TARGET crossvmaf_studio)
    set(development_files "$<TARGET_FILE:crossvmaf_studio>")
    foreach(target IN LISTS crossvmaf_package_runtime_targets)
        list(APPEND development_files "$<TARGET_FILE:${target}>")
    endforeach()
    add_custom_target(crossvmaf_studio_development ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${PROJECT_SOURCE_DIR}/out"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${development_files} "${PROJECT_SOURCE_DIR}/out"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "$<TARGET_FILE_DIR:crossvmaf_studio>/licenses" "${PROJECT_SOURCE_DIR}/out/licenses"
        DEPENDS crossvmaf_studio crossvmaf_ffmpeg_abi_check
        COMMAND_EXPAND_LISTS VERBATIM)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_custom_command(TARGET crossvmaf_studio_development POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:crossvmaf_ffmpeg>"
                "${PROJECT_SOURCE_DIR}/out/$<TARGET_SONAME_FILE_NAME:crossvmaf_ffmpeg>" VERBATIM)
    endif()
endif()
