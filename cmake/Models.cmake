function(crossvmaf_generate_model_pack target)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(model_names
        vmaf_v1.0.16_1d5h_2160.json
        vmaf_v1.0.16_3d0h_2160.json
        vmaf_v1.0.16_3d0h.json
        vmaf_v1.0.16_5d0h.json
        vmaf_v1.0.16_hfr_1d5h_2160.json
        vmaf_v1.0.16_hfr_3d0h_2160.json
        vmaf_v1.0.16_hfr_3d0h.json
        vmaf_v1.0.16_hfr_5d0h.json
    )
    set(model_files "")
    foreach(model_name IN LISTS model_names)
        list(APPEND model_files
            "${PROJECT_SOURCE_DIR}/models/source/${model_name}")
    endforeach()

    set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/model")
    set(output_file "${output_dir}/generated_model_pack.hpp")
    add_custom_command(
        OUTPUT "${output_file}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
        COMMAND "${Python3_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tools/model_pack/generate_model_pack.py"
                --output "${output_file}"
                ${model_files}
        DEPENDS
            "${PROJECT_SOURCE_DIR}/tools/model_pack/generate_model_pack.py"
            "${PROJECT_SOURCE_DIR}/models/MANIFEST.sha256"
            ${model_files}
        VERBATIM
        COMMENT "Verifying and packing the eight pinned VMAF 1.0.16 models"
    )

    target_sources(${target} PRIVATE "${output_file}")
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
endfunction()
