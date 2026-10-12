# Keep upstream's Meson source/assembly selection rather than maintaining a
# second dav1d build system. Only the resulting static library crosses into
# the parent build; each configuration has its own objects and MSVC runtime.
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m mesonbuild.mesonmain --version
    RESULT_VARIABLE crossvmaf_meson_result
    OUTPUT_QUIET ERROR_QUIET)
if(NOT crossvmaf_meson_result EQUAL 0)
    message(FATAL_ERROR
        "dav1d requires Meson: run '${Python3_EXECUTABLE} -m pip install meson ninja'")
endif()
if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "The dav1d integration currently requires a native build")
endif()

set(crossvmaf_dav1d_build_root "${CMAKE_BINARY_DIR}/_deps/crossvmaf-dav1d")
# Meson names static archives lib*.a, including those produced by MSVC lib.exe.
set(crossvmaf_dav1d_archive "libdav1d.a")
set(crossvmaf_dav1d_configuration "$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,Release>")
set(crossvmaf_dav1d_platform_arguments)
if(MSVC)
    # Clang/Ninja uses LLVM's native SDK discovery. Keep the environment
    # setup only for callers explicitly selecting a Visual Studio generator.
    if(CMAKE_GENERATOR_INSTANCE)
        list(APPEND crossvmaf_dav1d_platform_arguments
            --vcvars "${CMAKE_GENERATOR_INSTANCE}/VC/Auxiliary/Build/vcvars64.bat")
    endif()
    set(crossvmaf_dav1d_runtime "${CMAKE_MSVC_RUNTIME_LIBRARY}")
    if(NOT crossvmaf_dav1d_runtime)
        set(crossvmaf_dav1d_runtime "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
    list(APPEND crossvmaf_dav1d_platform_arguments
        --runtime "${crossvmaf_dav1d_runtime}")
endif()

add_custom_target(crossvmaf_dav1d_build
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/BuildDav1d.py"
        --source "${dav1d_SOURCE_DIR}"
        --build "${crossvmaf_dav1d_build_root}/${crossvmaf_dav1d_configuration}"
        --configuration "${crossvmaf_dav1d_configuration}"
        --compiler "${CMAKE_C_COMPILER}"
        --nasm "${CROSSVMAF_NASM_EXECUTABLE}"
        ${crossvmaf_dav1d_platform_arguments}
    BYPRODUCTS
        "${crossvmaf_dav1d_build_root}/${crossvmaf_dav1d_configuration}/src/${crossvmaf_dav1d_archive}"
    VERBATIM
    COMMENT "Building pinned dav1d with upstream Meson/Ninja")

add_library(crossvmaf_dav1d STATIC IMPORTED GLOBAL)
set_target_properties(crossvmaf_dav1d PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${dav1d_SOURCE_DIR}/include"
    IMPORTED_LOCATION
        "${crossvmaf_dav1d_build_root}/Release/src/${crossvmaf_dav1d_archive}")
set(crossvmaf_dav1d_configurations
    ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} Release)
list(REMOVE_DUPLICATES crossvmaf_dav1d_configurations)
foreach(configuration IN LISTS crossvmaf_dav1d_configurations)
    string(TOUPPER "${configuration}" configuration_upper)
    set_property(TARGET crossvmaf_dav1d APPEND PROPERTY
        IMPORTED_CONFIGURATIONS "${configuration}")
    set_target_properties(crossvmaf_dav1d PROPERTIES
        "IMPORTED_LOCATION_${configuration_upper}"
        "${crossvmaf_dav1d_build_root}/${configuration}/src/${crossvmaf_dav1d_archive}")
endforeach()
if(NOT WIN32)
    target_link_libraries(crossvmaf_dav1d INTERFACE
        Threads::Threads ${CMAKE_DL_LIBS} m)
endif()
add_dependencies(crossvmaf_dav1d crossvmaf_dav1d_build)
add_library(dav1d::dav1d ALIAS crossvmaf_dav1d)
