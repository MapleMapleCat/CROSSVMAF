if(NOT DEFINED INPUT_FILE OR NOT EXISTS "${INPUT_FILE}")
    message(FATAL_ERROR "EmbedBinary.cmake requires an existing INPUT_FILE")
endif()
if(NOT DEFINED OUTPUT_FILE OR NOT DEFINED SYMBOL_NAME)
    message(FATAL_ERROR "EmbedBinary.cmake requires OUTPUT_FILE and SYMBOL_NAME")
endif()

file(READ "${INPUT_FILE}" binary_hex HEX)
string(LENGTH "${binary_hex}" binary_hex_length)
math(EXPR binary_size "${binary_hex_length} / 2")
math(EXPR binary_remainder "${binary_size} % 4")
if(NOT binary_size GREATER 0 OR NOT binary_remainder EQUAL 0)
    message(FATAL_ERROR "SPIR-V binary size must be a non-zero multiple of four")
endif()

string(REGEX MATCHALL "[0-9a-fA-F][0-9a-fA-F]" binary_bytes "${binary_hex}")
set(initializers "")
set(column 0)
foreach(byte IN LISTS binary_bytes)
    string(APPEND initializers "0x${byte},")
    math(EXPR column "${column} + 1")
    if(column EQUAL 16)
        string(APPEND initializers "\n        ")
        set(column 0)
    else()
        string(APPEND initializers " ")
    endif()
endforeach()

file(WRITE "${OUTPUT_FILE}"
"#pragma once\n\n"
"#include <array>\n"
"#include <cstddef>\n"
"#include <cstdint>\n\n"
"namespace crossvmaf::gpu::generated {\n"
"alignas(4) inline constexpr std::array<std::uint8_t, ${binary_size}> ${SYMBOL_NAME}{\n"
"        ${initializers}\n"
"};\n"
"}  // namespace crossvmaf::gpu::generated\n")
