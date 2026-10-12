set(required_pipeline_variables
    GLSLANG_EXECUTABLE
    SPIRV_VAL_EXECUTABLE
    SPIRV_DIS_EXECUTABLE
    SOURCE_FILE
    INCLUDE_DIRECTORY
    SPV_FILE
    DISASSEMBLY_FILE
    HEADER_FILE
    SYMBOL_NAME
    AUDIT_SCRIPT
    EMBED_SCRIPT)

foreach(required_pipeline_variable IN LISTS required_pipeline_variables)
    if(NOT DEFINED ${required_pipeline_variable}
       OR "${${required_pipeline_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "RunShaderPipeline.cmake requires ${required_pipeline_variable}")
    endif()
endforeach()

set(temporary_spv_file "${SPV_FILE}.crossvmaf-tmp")
set(temporary_disassembly_file "${DISASSEMBLY_FILE}.crossvmaf-tmp")
set(temporary_header_file "${HEADER_FILE}.crossvmaf-tmp")

set(final_output_files
    "${SPV_FILE}"
    "${DISASSEMBLY_FILE}"
    "${HEADER_FILE}")
set(temporary_output_files
    "${temporary_spv_file}"
    "${temporary_disassembly_file}"
    "${temporary_header_file}")

function(remove_shader_pipeline_outputs)
    # A failed command must leave declared outputs missing so every generator
    # is forced to run the complete pipeline again on the next build.
    file(REMOVE ${final_output_files} ${temporary_output_files})
endfunction()

function(run_checked_shader_step step_description)
    execute_process(
        COMMAND ${ARGN}
        RESULT_VARIABLE shader_step_result)
    if(NOT shader_step_result STREQUAL "0")
        remove_shader_pipeline_outputs()
        message(FATAL_ERROR
            "${step_description} failed with result: ${shader_step_result}")
    endif()
endfunction()

function(promote_shader_output temporary_file final_file)
    file(RENAME "${temporary_file}" "${final_file}"
        RESULT promotion_result)
    if(NOT promotion_result STREQUAL "0")
        remove_shader_pipeline_outputs()
        message(FATAL_ERROR
            "Failed to promote ${temporary_file} to ${final_file}: "
            "${promotion_result}")
    endif()
endfunction()

foreach(required_input_file IN ITEMS
        "${GLSLANG_EXECUTABLE}"
        "${SPIRV_VAL_EXECUTABLE}"
        "${SPIRV_DIS_EXECUTABLE}"
        "${SOURCE_FILE}"
        "${AUDIT_SCRIPT}"
        "${EMBED_SCRIPT}")
    if(NOT EXISTS "${required_input_file}")
        remove_shader_pipeline_outputs()
        message(FATAL_ERROR
            "Shader pipeline input does not exist: ${required_input_file}")
    endif()
endforeach()

get_filename_component(shader_output_directory "${SPV_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${shader_output_directory}")
remove_shader_pipeline_outputs()

set(shader_compiler_command
    "${GLSLANG_EXECUTABLE}"
    --target-env vulkan1.1
    -V
    -S comp)

if(DEFINED SHADER_DEFINE_COUNT AND SHADER_DEFINE_COUNT GREATER 0)
    math(EXPR last_shader_definition_index "${SHADER_DEFINE_COUNT} - 1")
    foreach(shader_definition_index RANGE 0 ${last_shader_definition_index})
        set(shader_definition_variable
            "SHADER_DEFINE_${shader_definition_index}")
        if(NOT DEFINED ${shader_definition_variable})
            remove_shader_pipeline_outputs()
            message(FATAL_ERROR
                "Missing shader definition ${shader_definition_variable}")
        endif()
        list(APPEND shader_compiler_command
            "-D${${shader_definition_variable}}")
    endforeach()
endif()

list(APPEND shader_compiler_command
    "-I${INCLUDE_DIRECTORY}"
    -o "${temporary_spv_file}"
    "${SOURCE_FILE}")

run_checked_shader_step("Shader compilation" ${shader_compiler_command})
run_checked_shader_step(
    "SPIR-V validation"
    "${SPIRV_VAL_EXECUTABLE}"
    --target-env vulkan1.1
    "${temporary_spv_file}")
run_checked_shader_step(
    "SPIR-V disassembly"
    "${SPIRV_DIS_EXECUTABLE}"
    "${temporary_spv_file}"
    -o "${temporary_disassembly_file}")

set(shader_audit_command
    "${CMAKE_COMMAND}"
    "-DINPUT_FILE=${temporary_disassembly_file}"
    "-DEXPECTED_LOCAL_SIZE_X=${EXPECTED_LOCAL_SIZE_X}"
    "-DEXPECTED_UDIV_COUNT=${EXPECTED_UDIV_COUNT}"
    "-DMAX_UDIV_COUNT=${MAX_UDIV_COUNT}"
    "-DAUDIT_EXACT_WIDE_SHARED=${AUDIT_EXACT_WIDE_SHARED}"
    "-DAUDIT_COMPACT_COOPERATIVE=${AUDIT_COMPACT_COOPERATIVE}"
    "-DAUDIT_BLOCK_TRANSFORMED_SHARED=${AUDIT_BLOCK_TRANSFORMED_SHARED}"
    "-DAUDIT_BLOCK_SCALAR=${AUDIT_BLOCK_SCALAR}"
    "-DAUDIT_MEANS_WIDE_EXACT=${AUDIT_MEANS_WIDE_EXACT}"
    "-DAUDIT_MEANS_ROW_REDUCTION=${AUDIT_MEANS_ROW_REDUCTION}"
    "-DAUDIT_ADM3_FINALIZE_COOPERATIVE=${AUDIT_ADM3_FINALIZE_COOPERATIVE}"
    "-DAUDIT_CAMBI_CVALUE_STAGED=${AUDIT_CAMBI_CVALUE_STAGED}"
    "-DAUDIT_CAMBI_CVALUE_INT16=${AUDIT_CAMBI_CVALUE_INT16}"
    -P "${AUDIT_SCRIPT}")
run_checked_shader_step("SPIR-V audit" ${shader_audit_command})

run_checked_shader_step(
    "SPIR-V header generation"
    "${CMAKE_COMMAND}"
    "-DINPUT_FILE=${temporary_spv_file}"
    "-DOUTPUT_FILE=${temporary_header_file}"
    "-DSYMBOL_NAME=${SYMBOL_NAME}"
    -P "${EMBED_SCRIPT}")

foreach(required_temporary_output IN LISTS temporary_output_files)
    if(NOT EXISTS "${required_temporary_output}")
        remove_shader_pipeline_outputs()
        message(FATAL_ERROR
            "Shader pipeline did not create ${required_temporary_output}")
    endif()
endforeach()

# Commit the embedded header last. It is the source consumed by the host
# compiler and therefore acts as the completion marker for this transaction.
promote_shader_output(
    "${temporary_disassembly_file}" "${DISASSEMBLY_FILE}")
promote_shader_output("${temporary_spv_file}" "${SPV_FILE}")
promote_shader_output("${temporary_header_file}" "${HEADER_FILE}")
