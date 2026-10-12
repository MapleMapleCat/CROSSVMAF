if(NOT DEFINED INPUT_FILE OR NOT EXISTS "${INPUT_FILE}")
    message(FATAL_ERROR "GenerateSpirvManifest.cmake requires an existing INPUT_FILE")
endif()
if(NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "GenerateSpirvManifest.cmake requires OUTPUT_FILE")
endif()

file(STRINGS "${INPUT_FILE}" embedded_spirv_modules)
if(NOT embedded_spirv_modules)
    message(FATAL_ERROR "The embedded SPIR-V module list is empty")
endif()

list(SORT embedded_spirv_modules)
set(manifest_format "crossvmaf-embedded-spirv-manifest-v1")
set(canonical_manifest "${manifest_format}\n")
set(module_count 0)

foreach(module_record IN LISTS embedded_spirv_modules)
    if(NOT module_record MATCHES "^([^|]+)[|](.+)$")
        message(FATAL_ERROR "Invalid embedded SPIR-V module record: ${module_record}")
    endif()

    set(symbol_name "${CMAKE_MATCH_1}")
    set(spirv_file "${CMAKE_MATCH_2}")
    if(NOT EXISTS "${spirv_file}")
        message(FATAL_ERROR "Embedded SPIR-V module does not exist: ${spirv_file}")
    endif()

    file(SIZE "${spirv_file}" spirv_size)
    file(SHA256 "${spirv_file}" spirv_sha256)
    string(APPEND canonical_manifest
        "${symbol_name}\t${spirv_size}\t${spirv_sha256}\n")
    math(EXPR module_count "${module_count} + 1")
endforeach()

string(SHA256 aggregate_sha256 "${canonical_manifest}")

file(WRITE "${OUTPUT_FILE}"
"#pragma once\n\n"
"#include <cstdint>\n"
"#include <string_view>\n\n"
"namespace crossvmaf::gpu::generated {\n"
"inline constexpr std::string_view embedded_spirv_manifest_format =\n"
"    \"${manifest_format}\";\n"
"inline constexpr std::string_view embedded_spirv_manifest_sha256 =\n"
"    \"${aggregate_sha256}\";\n"
"inline constexpr std::uint32_t embedded_spirv_module_count = ${module_count}U;\n"
"}  // namespace crossvmaf::gpu::generated\n")
