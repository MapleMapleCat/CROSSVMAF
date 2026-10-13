cmake_minimum_required(VERSION 3.28)

project(
    CrossVMAF
    VERSION 0.1.2
    DESCRIPTION "Deterministic cross-vendor VMAF 1.0.16 execution engine"
    LANGUAGES C CXX
)

find_package(Threads REQUIRED)

option(CROSSVMAF_BUILD_TOOLS "Build CrossVMAF applications" ON)
option(CROSSVMAF_BUILD_STUDIO "Build the Slint-based CrossVMAF Studio desktop application" ON)
option(CROSSVMAF_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Version)
include(Dependencies)
include(DecodeBackends)
include(FFmpeg)
include(Licenses)
include(CompilerOptions)
include(Shaders)
include(Models)

add_subdirectory(src/numeric)
add_subdirectory(src/gpu)
add_subdirectory(src/video)
add_subdirectory(src/metrics)
add_subdirectory(src/scoring)
add_subdirectory(src/evaluation)
add_subdirectory(apps)
include(RuntimePackage)
include(CheckSourceDomains)
