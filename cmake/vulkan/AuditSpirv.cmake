if(NOT DEFINED INPUT_FILE OR NOT EXISTS "${INPUT_FILE}")
    message(FATAL_ERROR "AuditSpirv.cmake requires an existing INPUT_FILE")
endif()

file(READ "${INPUT_FILE}" spirv_text)

set(forbidden_patterns
    "OpTypeFloat"
    "OpTypeInt[ \t]+8"
    "OpTypeInt[ \t]+64"
    "OpCapability[ \t]+Int8"
    "OpCapability[ \t]+Int64"
    "OpCapability[ \t]+Float"
    "OpF(Add|Sub|Mul|Div|Rem|Mod|Negate|Convert)"
    "OpConvert[^\r\n]*F"
    "OpGroupNonUniform"
    "OpSubgroup"
    "OpExtension"
    "SPV_(NV|AMD|INTEL)"
)

if(NOT AUDIT_CAMBI_CVALUE_INT16)
    list(APPEND forbidden_patterns
        "OpTypeInt[ \t]+16"
        "OpCapability[ \t]+Int16")
endif()

foreach(pattern IN LISTS forbidden_patterns)
    if(spirv_text MATCHES "${pattern}")
        message(FATAL_ERROR
            "Forbidden SPIR-V construct '${pattern}' found in ${INPUT_FILE}")
    endif()
endforeach()

if(DEFINED EXPECTED_LOCAL_SIZE_X AND NOT EXPECTED_LOCAL_SIZE_X STREQUAL "")
    set(expected_local_size_pattern
        "OpExecutionMode[ \t]+%[^ \t\r\n]+[ \t]+LocalSize[ \t]+${EXPECTED_LOCAL_SIZE_X}[ \t]+1[ \t]+1")
    if(NOT spirv_text MATCHES "${expected_local_size_pattern}")
        message(FATAL_ERROR
            "Expected LocalSize ${EXPECTED_LOCAL_SIZE_X} 1 1 in ${INPUT_FILE}")
    endif()
endif()

if(DEFINED EXPECTED_UDIV_COUNT AND NOT EXPECTED_UDIV_COUNT STREQUAL "" AND
   DEFINED MAX_UDIV_COUNT AND NOT MAX_UDIV_COUNT STREQUAL "")
    message(FATAL_ERROR
        "EXPECTED_UDIV_COUNT and MAX_UDIV_COUNT are mutually exclusive")
endif()

if((DEFINED EXPECTED_UDIV_COUNT AND NOT EXPECTED_UDIV_COUNT STREQUAL "") OR
   (DEFINED MAX_UDIV_COUNT AND NOT MAX_UDIV_COUNT STREQUAL ""))
    string(REGEX MATCHALL "OpUDiv([ \t\r\n]|$)" unsigned_division_matches
        "${spirv_text}")
    list(LENGTH unsigned_division_matches unsigned_division_count)
endif()

if(DEFINED EXPECTED_UDIV_COUNT AND NOT EXPECTED_UDIV_COUNT STREQUAL "")
    if(NOT unsigned_division_count EQUAL ${EXPECTED_UDIV_COUNT})
        message(FATAL_ERROR
            "Expected ${EXPECTED_UDIV_COUNT} OpUDiv instructions, found "
            "${unsigned_division_count} in ${INPUT_FILE}")
    endif()
endif()

if(DEFINED MAX_UDIV_COUNT AND NOT MAX_UDIV_COUNT STREQUAL "")
    if(unsigned_division_count GREATER ${MAX_UDIV_COUNT})
        message(FATAL_ERROR
            "Allowed at most ${MAX_UDIV_COUNT} OpUDiv instructions, found "
            "${unsigned_division_count} in ${INPUT_FILE}")
    endif()
endif()

if(AUDIT_ADM3_FINALIZE_COOPERATIVE)
    string(REGEX MATCH
        "OpName[ \t]+(%[A-Za-z0-9_]+)[ \t]+\"finalize_roots\""
        finalize_roots_name "${spirv_text}")
    if(NOT finalize_roots_name)
        message(FATAL_ERROR
            "Expected named ADM3 finalize root storage in ${INPUT_FILE}")
    endif()
    set(finalize_roots_id "${CMAKE_MATCH_1}")
    if(NOT spirv_text MATCHES
       "${finalize_roots_id}[ \t]*=[ \t]*OpVariable[^\r\n]*[ \t]+Workgroup([ \t\r\n]|$)")
        message(FATAL_ERROR
            "ADM3 finalize roots are not Workgroup storage in ${INPUT_FILE}")
    endif()
    if(NOT spirv_text MATCHES "OpControlBarrier")
        message(FATAL_ERROR
            "Expected cooperative ADM3 finalize synchronization in ${INPUT_FILE}")
    endif()
endif()

if(AUDIT_CAMBI_CVALUE_STAGED)
    string(REGEX MATCH
        "OpName[ \t]+(%[A-Za-z0-9_]+)[ \t]+\"staged_source_rows\""
        staged_source_rows_name "${spirv_text}")
    if(NOT staged_source_rows_name)
        message(FATAL_ERROR
            "Expected named CAMBI staged rows in ${INPUT_FILE}")
    endif()
    set(staged_source_rows_id "${CMAKE_MATCH_1}")
    if(NOT spirv_text MATCHES
       "${staged_source_rows_id}[ \t]*=[ \t]*OpVariable[^\r\n]*[ \t]+Workgroup([ \t\r\n]|$)")
        message(FATAL_ERROR
            "CAMBI staged rows are not Workgroup storage in ${INPUT_FILE}")
    endif()
    if(NOT spirv_text MATCHES "OpControlBarrier")
        message(FATAL_ERROR
            "Expected CAMBI staged-row synchronization in ${INPUT_FILE}")
    endif()
endif()

if(AUDIT_CAMBI_CVALUE_INT16)
    if(NOT spirv_text MATCHES "OpCapability[ \t]+Int16")
        message(FATAL_ERROR
            "Expected Int16 capability in ${INPUT_FILE}")
    endif()
    if(NOT spirv_text MATCHES "OpTypeInt[ \t]+16[ \t]+0")
        message(FATAL_ERROR
            "Expected unsigned 16-bit histogram type in ${INPUT_FILE}")
    endif()
    string(REGEX MATCH
        "OpName[ \t]+(%[A-Za-z0-9_]+)[ \t]+\"histogram\""
        cambi_histogram_name "${spirv_text}")
    if(NOT cambi_histogram_name)
        message(FATAL_ERROR
            "Expected named CAMBI histogram in ${INPUT_FILE}")
    endif()
    set(cambi_histogram_id "${CMAKE_MATCH_1}")
    if(NOT spirv_text MATCHES
       "${cambi_histogram_id}[ \t]*=[ \t]*OpVariable[^\r\n]*[ \t]+Private([ \t\r\n]|$)")
        message(FATAL_ERROR
            "CAMBI Int16 histogram is not Private storage in ${INPUT_FILE}")
    endif()
endif()

if(AUDIT_COMPACT_COOPERATIVE)
    foreach(shared_value_name IN ITEMS
            compact_tail_scale
            compact_correction_scale
            compact_matrix_exponent)
        string(REGEX MATCH
            "OpName[ \t]+(%[A-Za-z0-9_]+)[ \t]+\"${shared_value_name}\""
            shared_value_match "${spirv_text}")
        if(NOT shared_value_match)
            message(FATAL_ERROR
                "Expected compact shared value ${shared_value_name} in ${INPUT_FILE}")
        endif()
        set(shared_value_id "${CMAKE_MATCH_1}")
        if(NOT spirv_text MATCHES
           "${shared_value_id}[ \t]*=[ \t]*OpVariable[^\r\n]*[ \t]+Workgroup([ \t\r\n]|$)")
            message(FATAL_ERROR
                "Compact value ${shared_value_name} is not Workgroup storage in ${INPUT_FILE}")
        endif()
    endforeach()
    if(NOT spirv_text MATCHES "OpAtomicOr")
        message(FATAL_ERROR
            "Expected cooperative compact failure propagation in ${INPUT_FILE}")
    endif()
    if(NOT spirv_text MATCHES "OpControlBarrier")
        message(FATAL_ERROR
            "Expected cooperative compact synchronization in ${INPUT_FILE}")
    endif()
endif()

if(AUDIT_BLOCK_TRANSFORMED_SHARED)
    foreach(shared_value_name IN ITEMS
            block_transformed_values
            block_phase_values)
        string(REGEX MATCH
            "OpName[ \t]+(%[A-Za-z0-9_]+)[ \t]+\"${shared_value_name}\""
            shared_value_match "${spirv_text}")
        if(NOT shared_value_match)
            message(FATAL_ERROR
                "Expected named block-statistics shared value "
                "${shared_value_name} in ${INPUT_FILE}")
        endif()
        set(shared_value_id "${CMAKE_MATCH_1}")
        if(NOT spirv_text MATCHES
           "${shared_value_id}[ \t]*=[ \t]*OpVariable[^\r\n]*[ \t]+Workgroup([ \t\r\n]|$)")
            message(FATAL_ERROR
                "Block-statistics value ${shared_value_name} is not "
                "Workgroup storage in ${INPUT_FILE}")
        endif()
    endforeach()
    if(NOT spirv_text MATCHES "OpControlBarrier")
        message(FATAL_ERROR
            "Expected cooperative block-statistics synchronization in "
            "${INPUT_FILE}")
    endif()
endif()

if(AUDIT_BLOCK_SCALAR)
    if(spirv_text MATCHES
       "OpVariable[^\r\n]*[ \t]+Workgroup([ \t\r\n]|$)")
        message(FATAL_ERROR
            "Block-statistics scalar module uses Workgroup storage in "
            "${INPUT_FILE}")
    endif()
    if(spirv_text MATCHES "OpControlBarrier")
        message(FATAL_ERROR
            "Block-statistics scalar module contains a workgroup barrier in "
            "${INPUT_FILE}")
    endif()
    if(NOT spirv_text MATCHES "OpAtomicOr")
        message(FATAL_ERROR
            "Expected scalar block-statistics failure propagation in "
            "${INPUT_FILE}")
    endif()
endif()

if(AUDIT_MEANS_WIDE_EXACT)
    if(NOT spirv_text MATCHES
       "OpName[ \t]+%[A-Za-z0-9_]+[ \t]+\"exact_(row_)?sum\"")
        message(FATAL_ERROR
            "Expected named exact Means accumulator in ${INPUT_FILE}")
    endif()
    if(NOT spirv_text MATCHES
       "OpTypeVector[ \t]+%uint[ \t]+4([ \t\r\n]|$)")
        message(FATAL_ERROR
            "Expected at least 96-bit exact Means accumulator storage in ${INPUT_FILE}")
    endif()
    if(NOT spirv_text MATCHES "OpControlBarrier")
        message(FATAL_ERROR
            "Expected uniform M2 staging synchronization in ${INPUT_FILE}")
    endif()
endif()

if(AUDIT_MEANS_ROW_REDUCTION)
    string(REGEX MATCH
        "OpName[ \t]+(%[A-Za-z0-9_]+)[ \t]+\"means_exact_row_sums\""
        means_exact_row_sums_name "${spirv_text}")
    if(NOT means_exact_row_sums_name)
        message(FATAL_ERROR
            "Expected named M3 exact row-sum storage in ${INPUT_FILE}")
    endif()
    set(means_exact_row_sums_id "${CMAKE_MATCH_1}")
    if(NOT spirv_text MATCHES
       "${means_exact_row_sums_id}[ \t]*=[ \t]*OpVariable[^\r\n]*[ \t]+Workgroup([ \t\r\n]|$)")
        message(FATAL_ERROR
            "M3 exact row sums are not Workgroup storage in ${INPUT_FILE}")
    endif()
    if(NOT spirv_text MATCHES "OpAtomicOr")
        message(FATAL_ERROR
            "Expected M3 cooperative failure propagation in ${INPUT_FILE}")
    endif()
    if(NOT spirv_text MATCHES "OpControlBarrier")
        message(FATAL_ERROR
            "Expected M3 uniform tree-reduction barriers in ${INPUT_FILE}")
    endif()
endif()

if(AUDIT_EXACT_WIDE_SHARED)
    string(REGEX MATCH
        "(%[A-Za-z0-9_]+)[ \t]*=[ \t]*OpConstant[ \t]+%uint[ \t]+80([ \t\r\n]|$)"
        wide_limb_count_constant "${spirv_text}")
    if(NOT wide_limb_count_constant)
        message(FATAL_ERROR
            "Expected the exact 80-limb count in ${INPUT_FILE}")
    endif()
    set(wide_limb_count_id "${CMAKE_MATCH_1}")
    if(spirv_text MATCHES
       "OpTypeArray[ \t]+%uint[ \t]+${wide_limb_count_id}([ \t\r\n]|$)")
        message(FATAL_ERROR
            "Exact module contains a private-style uint[80] array in ${INPUT_FILE}")
    endif()

    string(REGEX MATCH
        "OpName[ \t]+(%[A-Za-z0-9_]+)[ \t]+\"exact_wide_limbs\""
        exact_wide_limbs_name "${spirv_text}")
    if(NOT exact_wide_limbs_name)
        message(FATAL_ERROR
            "Expected named exact wide-limb storage in ${INPUT_FILE}")
    endif()
    set(exact_wide_limbs_id "${CMAKE_MATCH_1}")
    if(NOT spirv_text MATCHES
       "${exact_wide_limbs_id}[ \t]*=[ \t]*OpVariable[^\r\n]*[ \t]+Workgroup([ \t\r\n]|$)")
        message(FATAL_ERROR
            "Exact wide-limb storage is not Workgroup storage in ${INPUT_FILE}")
    endif()

    string(REGEX MATCH
        "(%[A-Za-z0-9_]+)[ \t]*=[ \t]*OpConstant[ \t]+%uint[ \t]+400([ \t\r\n]|$)"
        shared_limb_word_count_constant "${spirv_text}")
    if(NOT shared_limb_word_count_constant)
        message(FATAL_ERROR
            "Expected the 400-word exact shared SoA in ${INPUT_FILE}")
    endif()
    set(shared_limb_word_count_id "${CMAKE_MATCH_1}")
    if(NOT spirv_text MATCHES
       "OpTypeArray[ \t]+%uint[ \t]+${shared_limb_word_count_id}([ \t\r\n]|$)")
        message(FATAL_ERROR
            "Expected a uint[400] exact shared SoA in ${INPUT_FILE}")
    endif()

    string(REGEX MATCH
        "OpName[ \t]+(%[A-Za-z0-9_]+)[ \t]+\"exact_wide_workspace\""
        exact_wide_workspace_name "${spirv_text}")
    if(NOT exact_wide_workspace_name)
        message(FATAL_ERROR
            "Expected named exact cooperative workspace in ${INPUT_FILE}")
    endif()
    set(exact_wide_workspace_id "${CMAKE_MATCH_1}")
    if(NOT spirv_text MATCHES
       "${exact_wide_workspace_id}[ \t]*=[ \t]*OpVariable[^\r\n]*[ \t]+Workgroup([ \t\r\n]|$)")
        message(FATAL_ERROR
            "Exact cooperative workspace is not Workgroup storage in ${INPUT_FILE}")
    endif()

    string(REGEX MATCH
        "(%[A-Za-z0-9_]+)[ \t]*=[ \t]*OpConstant[ \t]+%uint[ \t]+384([ \t\r\n]|$)"
        shared_workspace_word_count_constant "${spirv_text}")
    if(NOT shared_workspace_word_count_constant)
        message(FATAL_ERROR
            "Expected the 384-word exact cooperative workspace in ${INPUT_FILE}")
    endif()
    set(shared_workspace_word_count_id "${CMAKE_MATCH_1}")
    if(NOT spirv_text MATCHES
       "OpTypeArray[ \t]+%uint[ \t]+${shared_workspace_word_count_id}([ \t\r\n]|$)")
        message(FATAL_ERROR
            "Expected a uint[384] exact cooperative workspace in ${INPUT_FILE}")
    endif()

    if(NOT spirv_text MATCHES "OpAtomicIAdd")
        message(FATAL_ERROR
            "Expected cooperative exact product accumulation in ${INPUT_FILE}")
    endif()
    if(NOT spirv_text MATCHES "OpControlBarrier")
        message(FATAL_ERROR
            "Expected workgroup synchronization in ${INPUT_FILE}")
    endif()
endif()
