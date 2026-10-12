function(crossvmaf_compile_shader target source_file symbol_name)
    set(shader_tool_directory
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/vulkan")
    cmake_parse_arguments(
        SHADER
        "AUDIT_EXACT_WIDE_SHARED;AUDIT_COMPACT_COOPERATIVE;AUDIT_BLOCK_TRANSFORMED_SHARED;AUDIT_BLOCK_SCALAR;AUDIT_MEANS_WIDE_EXACT;AUDIT_MEANS_ROW_REDUCTION;AUDIT_ADM3_FINALIZE_COOPERATIVE;AUDIT_CAMBI_CVALUE_STAGED;AUDIT_CAMBI_CVALUE_INT16"
        "OUTPUT_NAME;EXPECTED_LOCAL_SIZE_X;EXPECTED_UDIV_COUNT;MAX_UDIV_COUNT"
        "DEFINES;DEPENDS"
        ${ARGN})

    get_filename_component(shader_name "${source_file}" NAME_WE)
    if(SHADER_OUTPUT_NAME)
        set(shader_name "${SHADER_OUTPUT_NAME}")
    endif()
    set(shader_definition_script_arguments)
    set(shader_definition_count 0)
    foreach(shader_definition IN LISTS SHADER_DEFINES)
        list(APPEND shader_definition_script_arguments
            "-DSHADER_DEFINE_${shader_definition_count}=${shader_definition}")
        math(EXPR shader_definition_count "${shader_definition_count} + 1")
    endforeach()
    set(shader_output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders")
    set(spv_file "${shader_output_dir}/${shader_name}.spv")
    set(disassembly_file "${shader_output_dir}/${shader_name}.spvasm")
    set(header_file "${shader_output_dir}/${shader_name}_spv.hpp")

    add_custom_command(
        OUTPUT "${header_file}" "${spv_file}" "${disassembly_file}"
        COMMAND "${CMAKE_COMMAND}"
                "-DGLSLANG_EXECUTABLE=$<TARGET_FILE:glslang-standalone>"
                "-DSPIRV_VAL_EXECUTABLE=$<TARGET_FILE:spirv-val>"
                "-DSPIRV_DIS_EXECUTABLE=$<TARGET_FILE:spirv-dis>"
                "-DSOURCE_FILE=${PROJECT_SOURCE_DIR}/${source_file}"
                "-DINCLUDE_DIRECTORY=${PROJECT_SOURCE_DIR}/src/numeric"
                "-DSPV_FILE=${spv_file}"
                "-DDISASSEMBLY_FILE=${disassembly_file}"
                "-DHEADER_FILE=${header_file}"
                "-DSYMBOL_NAME=${symbol_name}"
                "-DAUDIT_SCRIPT=${shader_tool_directory}/AuditSpirv.cmake"
                "-DEMBED_SCRIPT=${shader_tool_directory}/EmbedBinary.cmake"
                "-DEXPECTED_LOCAL_SIZE_X=${SHADER_EXPECTED_LOCAL_SIZE_X}"
                "-DEXPECTED_UDIV_COUNT=${SHADER_EXPECTED_UDIV_COUNT}"
                "-DMAX_UDIV_COUNT=${SHADER_MAX_UDIV_COUNT}"
                "-DAUDIT_EXACT_WIDE_SHARED=${SHADER_AUDIT_EXACT_WIDE_SHARED}"
                "-DAUDIT_COMPACT_COOPERATIVE=${SHADER_AUDIT_COMPACT_COOPERATIVE}"
                "-DAUDIT_BLOCK_TRANSFORMED_SHARED=${SHADER_AUDIT_BLOCK_TRANSFORMED_SHARED}"
                "-DAUDIT_BLOCK_SCALAR=${SHADER_AUDIT_BLOCK_SCALAR}"
                "-DAUDIT_MEANS_WIDE_EXACT=${SHADER_AUDIT_MEANS_WIDE_EXACT}"
                "-DAUDIT_MEANS_ROW_REDUCTION=${SHADER_AUDIT_MEANS_ROW_REDUCTION}"
                "-DAUDIT_ADM3_FINALIZE_COOPERATIVE=${SHADER_AUDIT_ADM3_FINALIZE_COOPERATIVE}"
                "-DAUDIT_CAMBI_CVALUE_STAGED=${SHADER_AUDIT_CAMBI_CVALUE_STAGED}"
                "-DAUDIT_CAMBI_CVALUE_INT16=${SHADER_AUDIT_CAMBI_CVALUE_INT16}"
                "-DSHADER_DEFINE_COUNT=${shader_definition_count}"
                ${shader_definition_script_arguments}
                -P "${shader_tool_directory}/RunShaderPipeline.cmake"
        DEPENDS
            "${PROJECT_SOURCE_DIR}/${source_file}"
            ${SHADER_DEPENDS}
            "${shader_tool_directory}/AuditSpirv.cmake"
            "${shader_tool_directory}/EmbedBinary.cmake"
            "${shader_tool_directory}/RunShaderPipeline.cmake"
            "${PROJECT_SOURCE_DIR}/src/numeric/glsl/integer.glsl"
            "${PROJECT_SOURCE_DIR}/src/numeric/glsl/q32.glsl"
            "${PROJECT_SOURCE_DIR}/src/numeric/glsl/q48_linear.glsl"
            "${PROJECT_SOURCE_DIR}/src/numeric/glsl/binary32_quantize.glsl"
            "${PROJECT_SOURCE_DIR}/src/numeric/glsl/transcendentals.glsl"
            glslang-standalone
            spirv-val
            spirv-dis
        VERBATIM
        COMMENT "Compiling and auditing integer-only shader ${source_file}"
    )

    target_sources(${target} PRIVATE "${header_file}")
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

    # Keep the compiled module registry coupled to the command that produces
    # each SPIR-V binary. The registry is finalized into the persistent
    # pipeline-cache identity after every Vulkan shader has been declared.
    set_property(
        GLOBAL APPEND PROPERTY CROSSVMAF_EMBEDDED_SPIRV_MODULES
        "${symbol_name}|${spv_file}")
endfunction()

function(crossvmaf_generate_spirv_manifest target)
    set(shader_tool_directory
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/vulkan")
    get_property(
        embedded_spirv_modules
        GLOBAL PROPERTY CROSSVMAF_EMBEDDED_SPIRV_MODULES)
    if(NOT embedded_spirv_modules)
        message(FATAL_ERROR "No embedded SPIR-V modules were registered")
    endif()

    list(SORT embedded_spirv_modules)
    set(shader_output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders")
    set(module_list_file "${shader_output_dir}/embedded_spirv_manifest.modules")
    set(manifest_header "${shader_output_dir}/embedded_spirv_manifest.hpp")

    string(REPLACE ";" "\n" module_list_contents "${embedded_spirv_modules}")
    file(MAKE_DIRECTORY "${shader_output_dir}")
    file(WRITE "${module_list_file}" "${module_list_contents}\n")

    set(spirv_dependencies)
    foreach(module_record IN LISTS embedded_spirv_modules)
        string(REGEX REPLACE "^[^|]+[|]" "" spv_file "${module_record}")
        list(APPEND spirv_dependencies "${spv_file}")
    endforeach()

    add_custom_command(
        OUTPUT "${manifest_header}"
        COMMAND "${CMAKE_COMMAND}"
                "-DINPUT_FILE=${module_list_file}"
                "-DOUTPUT_FILE=${manifest_header}"
                -P "${shader_tool_directory}/GenerateSpirvManifest.cmake"
        DEPENDS
            "${module_list_file}"
            "${shader_tool_directory}/GenerateSpirvManifest.cmake"
            ${spirv_dependencies}
        VERBATIM
        COMMENT "Generating embedded SPIR-V pipeline-cache identity"
    )

    target_sources(${target} PRIVATE "${manifest_header}")
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
endfunction()
