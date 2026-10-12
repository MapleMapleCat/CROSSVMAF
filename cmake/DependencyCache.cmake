include_guard(GLOBAL)

# Set the default before FetchContent initializes its cache entry. Explicit
# -DFETCHCONTENT_BASE_DIR overrides remain supported, including without presets.
set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/.cache/fetchcontent"
    CACHE PATH "Shared third-party source cache")
include(FetchContent)
find_package(Git REQUIRED)

function(crossvmaf_fetchcontent_declare name)
    cmake_parse_arguments(PARSE_ARGV 1 dependency "" "GIT_REPOSITORY;GIT_TAG" "")
    if(NOT dependency_GIT_REPOSITORY OR NOT dependency_GIT_TAG)
        message(FATAL_ERROR "${name}: cached dependencies require a pinned Git repository and tag")
    endif()

    string(TOLOWER "${name}" lower_name)
    string(TOUPPER "${name}" upper_name)
    # Keep automatic hits out of CMakeCache.txt so a later tag change cannot
    # mistake a previous cache hit for an explicit user source override.
    set(FETCHCONTENT_SOURCE_DIR_${upper_name} "" CACHE PATH
        "Explicit source override for ${name}")
    string(SHA256 identity "${dependency_GIT_REPOSITORY}\n${dependency_GIT_TAG}")
    string(SUBSTRING "${identity}" 0 12 identity)
    set(source_dir "${FETCHCONTENT_BASE_DIR}/${lower_name}-${identity}-src")
    set(ready_stamp "${source_dir}/.crossvmaf-populated")

    # Download completion is independent of the generator/compiler-specific
    # subbuild stamps. A new build tree can therefore reuse a complete checkout
    # without letting ExternalProject delete it and clone it again.
    if(NOT FETCHCONTENT_SOURCE_DIR_${upper_name} AND EXISTS "${ready_stamp}")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse --verify HEAD
            RESULT_VARIABLE head_result OUTPUT_VARIABLE head
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}"
                rev-parse --verify "${dependency_GIT_TAG}^{commit}"
            RESULT_VARIABLE tag_result OUTPUT_VARIABLE pinned_commit
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(head_result EQUAL 0 AND tag_result EQUAL 0 AND head STREQUAL pinned_commit)
            set(FETCHCONTENT_SOURCE_DIR_${upper_name} "${source_dir}" PARENT_SCOPE)
            message(STATUS "Reusing cached ${name}: ${source_dir}")
        endif()
    endif()

    FetchContent_Declare(${name}
        ${ARGN}
        SOURCE_DIR "${source_dir}"
        # Run only after Git checkout and recursive submodule population succeed.
        PATCH_COMMAND "${CMAKE_COMMAND}" -E touch "${ready_stamp}"
    )
endfunction()
