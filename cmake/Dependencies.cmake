include(DependencyCache)

set(FETCHCONTENT_QUIET OFF)

# Dependency versions are part of the CrossVMAF determinism contract. Keep
# these tags synchronized with docs/DEPENDENCIES.md and the conformance report.
# Only versioned sources are shared through FETCHCONTENT_BASE_DIR. Compiler-specific
# CMake caches and objects must stay in the current build tree.
set(CROSSVMAF_VULKAN_SDK_TAG "vulkan-sdk-1.4.350.1")
set(CROSSVMAF_VOLK_TAG "1.4.350")
set(CROSSVMAF_GLSLANG_TAG "16.4.0")
set(CROSSVMAF_DAV1D_TAG "1.5.4")
set(CROSSVMAF_SLINT_TAG "v1.17.1")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$")
    find_program(CROSSVMAF_NASM_EXECUTABLE
        NAMES nasm nasm.exe
        HINTS "$ENV{ProgramFiles}/NASM" "$ENV{LOCALAPPDATA}/bin/NASM"
        REQUIRED
    )
endif()
crossvmaf_fetchcontent_declare(
    dav1d
    BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/dav1d-build"
    SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/dav1d-subbuild"
    GIT_REPOSITORY https://code.videolan.org/videolan/dav1d.git
    GIT_TAG ${CROSSVMAF_DAV1D_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dav1d)
include("${CMAKE_CURRENT_LIST_DIR}/Dav1d.cmake")

crossvmaf_fetchcontent_declare(
    VulkanHeaders
    BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/vulkanheaders-build"
    SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/vulkanheaders-subbuild"
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG ${CROSSVMAF_VULKAN_SDK_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(VulkanHeaders)

crossvmaf_fetchcontent_declare(
    volk
    BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/volk-build"
    SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/volk-subbuild"
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG ${CROSSVMAF_VOLK_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(volk)

# Runtime shaders are required; always build their compiler and validation tools.
set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_CTEST OFF CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
crossvmaf_fetchcontent_declare(
    glslang
    BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/glslang-build"
    SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/glslang-subbuild"
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG ${CROSSVMAF_GLSLANG_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glslang)

crossvmaf_fetchcontent_declare(
    spirv_headers
    BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/spirv_headers-build"
    SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/spirv_headers-subbuild"
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG ${CROSSVMAF_VULKAN_SDK_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spirv_headers)
set(SPIRV-Headers_SOURCE_DIR "${spirv_headers_SOURCE_DIR}" CACHE PATH "" FORCE)

set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES OFF CACHE BOOL "" FORCE)
set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
crossvmaf_fetchcontent_declare(
    spirv_tools
    BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/spirv_tools-build"
    SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/spirv_tools-subbuild"
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
    GIT_TAG ${CROSSVMAF_VULKAN_SDK_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spirv_tools)

crossvmaf_fetchcontent_declare(
    nlohmann_json
    BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/nlohmann_json-build"
    SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/nlohmann_json-subbuild"
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

if(CROSSVMAF_BUILD_STUDIO)
    # Slint's default Corrosion subbuild otherwise shares a generator-specific
    # CMake cache with other presets using the same source download cache.
    crossvmaf_fetchcontent_declare(Corrosion
        BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/corrosion-build"
        SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/corrosion-subbuild"
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG v0.6.1 GIT_SHALLOW TRUE)
    # Build both the runtime and UI compiler from the same pinned source tag.
    # An empty SLINT_COMPILER prevents reuse/download of a prebuilt compiler.
    set(SLINT_COMPILER "" CACHE STRING "Build the Slint compiler from source" FORCE)
    set(SLINT_BUILD_RUNTIME ON CACHE BOOL "" FORCE)
    set(SLINT_FEATURE_COMPILER ON CACHE BOOL "" FORCE)
    set(SLINT_FEATURE_BACKEND_QT OFF CACHE BOOL "" FORCE)
    set(SLINT_FEATURE_BACKEND_WINIT ON CACHE BOOL "" FORCE)
    set(SLINT_FEATURE_RENDERER_FEMTOVG ON CACHE BOOL "" FORCE)
    set(SLINT_FEATURE_RENDERER_SKIA OFF CACHE BOOL "" FORCE)
    crossvmaf_fetchcontent_declare(
        slint
        BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/slint-build"
        SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/slint-subbuild"
        GIT_REPOSITORY https://github.com/slint-ui/slint.git
        GIT_TAG ${CROSSVMAF_SLINT_TAG}
        GIT_SHALLOW TRUE
    )
    # Slint defaults BUILD_SHARED_LIBS to ON. Keep that default local so a
    # later configure does not silently turn glslang into a shared library.
    block(SCOPE_FOR VARIABLES)
        if(NOT DEFINED BUILD_SHARED_LIBS)
            set(BUILD_SHARED_LIBS ON)
        endif()
        FetchContent_MakeAvailable(slint)
    endblock()
    if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # Slint 1.17.1 selects the GNU driver flag for all Clang variants.
        # clang-cl needs the MSVC spelling instead; leave upstream sources intact.
        get_target_property(crossvmaf_slint_compile_options Slint INTERFACE_COMPILE_OPTIONS)
        list(REMOVE_ITEM crossvmaf_slint_compile_options
            "$<$<CXX_COMPILER_ID:Clang>:-Wa,-mbig-obj>")
        set_property(TARGET Slint PROPERTY INTERFACE_COMPILE_OPTIONS
            ${crossvmaf_slint_compile_options} /bigobj)
    endif()
endif()
