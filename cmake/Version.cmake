# PROJECT_VERSION is the only manually maintained software release version.
set(CROSSVMAF_VERSION_INCLUDE_DIRECTORY "${PROJECT_BINARY_DIR}/generated/include")
set(CROSSVMAF_VERSION_SLINT "${PROJECT_BINARY_DIR}/generated/software-version.slint")

configure_file("${CMAKE_CURRENT_LIST_DIR}/version.hpp.in"
    "${CROSSVMAF_VERSION_INCLUDE_DIRECTORY}/crossvmaf/version.hpp" @ONLY)
configure_file("${CMAKE_CURRENT_LIST_DIR}/version.slint.in"
    "${CROSSVMAF_VERSION_SLINT}" @ONLY)

add_library(crossvmaf_version INTERFACE)
add_library(CrossVMAF::version ALIAS crossvmaf_version)
target_include_directories(crossvmaf_version INTERFACE
    "$<BUILD_INTERFACE:${CROSSVMAF_VERSION_INCLUDE_DIRECTORY}>")
target_compile_features(crossvmaf_version INTERFACE cxx_std_17)
