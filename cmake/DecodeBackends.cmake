option(CROSSVMAF_ENABLE_VULKAN_VIDEO "Enable Vulkan Video decoding" ON)
option(CROSSVMAF_ENABLE_QSV "Enable Intel VPL/QSV decoding" ON)
option(CROSSVMAF_ENABLE_NVDEC "Enable NVIDIA NVDEC decoding" ON)
option(CROSSVMAF_ENABLE_VAAPI "Enable Linux VA-API decoding" OFF)
if(CROSSVMAF_ENABLE_VAAPI AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The VA-API decode backend requires Linux")
endif()
set(crossvmaf_decode_definitions)
foreach(backend VULKAN_VIDEO QSV NVDEC VAAPI)
    list(APPEND crossvmaf_decode_definitions
        "CROSSVMAF_ENABLE_${backend}=$<BOOL:${CROSSVMAF_ENABLE_${backend}}>")
endforeach()
set(crossvmaf_decode_configure_arguments)
set(crossvmaf_decode_codec_sources)
set(crossvmaf_decode_util_sources)
set(crossvmaf_decode_required_components)
if(CROSSVMAF_ENABLE_VULKAN_VIDEO)
    list(APPEND crossvmaf_decode_configure_arguments --enable-hwaccel=h264_vulkan,hevc_vulkan,av1_vulkan,vp9_vulkan)
    foreach(codec H264 HEVC AV1 VP9)
        list(APPEND crossvmaf_decode_required_components CONFIG_${codec}_VULKAN_HWACCEL)
    endforeach()
endif()
if(CROSSVMAF_ENABLE_NVDEC)
    crossvmaf_fetchcontent_declare(crossvmaf_nvcodec
        BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/nvcodec-build"
        SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/nvcodec-subbuild"
        GIT_REPOSITORY https://github.com/FFmpeg/nv-codec-headers.git
        GIT_TAG 1889e62e2d35ff7aa9baca2bceb14f053785e6f1
        SOURCE_SUBDIR crossvmaf-no-cmake)
    FetchContent_MakeAvailable(crossvmaf_nvcodec)
    list(APPEND crossvmaf_decode_configure_arguments --enable-ffnvcodec --enable-nvdec
        --enable-hwaccel=h264_nvdec,hevc_nvdec,av1_nvdec,vp9_nvdec)
    list(APPEND crossvmaf_decode_util_sources libavutil/hwcontext_cuda.c)
    list(APPEND crossvmaf_decode_codec_sources libavcodec/nvdec.c
        libavcodec/nvdec_h264.c libavcodec/nvdec_hevc.c libavcodec/nvdec_av1.c libavcodec/nvdec_vp9.c)
    foreach(codec H264 HEVC AV1 VP9)
        list(APPEND crossvmaf_decode_required_components CONFIG_${codec}_NVDEC_HWACCEL)
    endforeach()
endif()
if(CROSSVMAF_ENABLE_QSV)
    crossvmaf_fetchcontent_declare(crossvmaf_vpl
        BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/vpl-build"
        SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/vpl-subbuild"
        GIT_REPOSITORY https://github.com/intel/libvpl.git
        GIT_TAG c45b5d786bf7cdabbe49ff1bab78693ad78feb78
        SOURCE_SUBDIR crossvmaf-no-cmake)
    FetchContent_MakeAvailable(crossvmaf_vpl)
    # FFmpeg configure links MFXLoad. Build the pinned dispatcher before that
    # probe; its GPU implementation is still selected dynamically at runtime.
    set(crossvmaf_vpl_build "${CMAKE_BINARY_DIR}/_deps/vpl-dispatcher")
    set(crossvmaf_vpl_install "${CMAKE_BINARY_DIR}/_deps/vpl-install")
    set(crossvmaf_vpl_tool_arguments)
    foreach(tool CMAKE_AR CMAKE_LINKER CMAKE_RC_COMPILER CMAKE_MT)
        if(${tool})
            list(APPEND crossvmaf_vpl_tool_arguments "-D${tool}:FILEPATH=${${tool}}")
        endif()
    endforeach()
    set(crossvmaf_vpl_cache_arguments)
    if(EXISTS "${crossvmaf_vpl_build}/CMakeCache.txt")
        load_cache("${crossvmaf_vpl_build}" READ_WITH_PREFIX previous_vpl_
            CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_AR CMAKE_LINKER CMAKE_RC_COMPILER CMAKE_MT)
        foreach(tool CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_AR CMAKE_LINKER CMAKE_RC_COMPILER CMAKE_MT)
            if(${tool} AND NOT "${${tool}}" STREQUAL "${previous_vpl_${tool}}")
                # CMake's generated RC compiler metadata otherwise keeps the
                # old tool even when the cache command-line value changes.
                set(crossvmaf_vpl_cache_arguments --fresh)
            endif()
        endforeach()
    endif()
    if(MSVC)
        # The pinned VPL version.rc contains a Windows-1252 registered sign.
        # Make its encoding explicit for llvm-rc, independent of host locale.
        list(APPEND crossvmaf_vpl_tool_arguments "-DCMAKE_RC_FLAGS:STRING=/C1252")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -S "${crossvmaf_vpl_SOURCE_DIR}" -B "${crossvmaf_vpl_build}" -G Ninja
        ${crossvmaf_vpl_cache_arguments}
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}" "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        ${crossvmaf_vpl_tool_arguments}
        -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=${crossvmaf_vpl_install}"
        -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DINSTALL_EXAMPLES=OFF
        -DBUILD_EXPERIMENTAL=OFF -DCMAKE_INSTALL_LIBDIR=lib
        RESULT_VARIABLE vpl_configure OUTPUT_FILE "${CMAKE_BINARY_DIR}/vpl-configure.log" ERROR_FILE "${CMAKE_BINARY_DIR}/vpl-configure-error.log")
    if(NOT vpl_configure EQUAL 0)
        message(FATAL_ERROR "VPL configure failed; see vpl-configure-error.log")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" --build "${crossvmaf_vpl_build}" --target install --parallel 8
        RESULT_VARIABLE vpl_build OUTPUT_FILE "${CMAKE_BINARY_DIR}/vpl-build.log" ERROR_FILE "${CMAKE_BINARY_DIR}/vpl-build-error.log")
    if(NOT vpl_build EQUAL 0)
        message(FATAL_ERROR "VPL build failed; see vpl-build.log / vpl-build-error.log")
    endif()
    block(SCOPE_FOR VARIABLES)
        set(VPL_SHARED ON)
        set(VPL_DIR "${crossvmaf_vpl_install}/lib/cmake/vpl")
        find_package(VPL CONFIG REQUIRED COMPONENTS dispatcher NO_DEFAULT_PATH
            PATHS "${VPL_DIR}")
    endblock()
    add_library(CrossVMAF::vpl ALIAS VPL::dispatcher)
    # The external C dispatcher is built in Release, also for Debug clients.
    set_target_properties(VPL::dispatcher PROPERTIES
        IMPORTED_CONFIGURATIONS Release MAP_IMPORTED_CONFIG_DEBUG Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    list(APPEND crossvmaf_decode_configure_arguments --enable-libvpl --enable-decoder=h264_qsv,hevc_qsv,av1_qsv,vp9_qsv)
    list(APPEND crossvmaf_decode_util_sources libavutil/hwcontext_qsv.c)
    list(APPEND crossvmaf_decode_codec_sources libavcodec/qsv.c libavcodec/qsvdec.c libavcodec/atsc_a53.c
        libavcodec/bsf/h264_mp4toannexb.c libavcodec/bsf/hevc_mp4toannexb.c)
    foreach(codec H264 HEVC AV1 VP9)
        list(APPEND crossvmaf_decode_required_components CONFIG_${codec}_QSV_DECODER)
    endforeach()
    if(WIN32)
        list(APPEND crossvmaf_decode_configure_arguments --enable-d3d11va)
        list(APPEND crossvmaf_decode_util_sources libavutil/hwcontext_d3d11va.c)
        list(APPEND crossvmaf_decode_codec_sources libavcodec/dxva2.c)
    endif()
endif()
if(CROSSVMAF_ENABLE_VAAPI OR (CROSSVMAF_ENABLE_QSV AND CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(CROSSVMAF_VA REQUIRED IMPORTED_TARGET libva libva-drm libdrm)
    list(APPEND crossvmaf_decode_configure_arguments --enable-vaapi --enable-libdrm)
    list(APPEND crossvmaf_decode_util_sources libavutil/hwcontext_vaapi.c libavutil/hwcontext_drm.c)
    if(CROSSVMAF_ENABLE_VAAPI)
        list(APPEND crossvmaf_decode_configure_arguments --enable-hwaccel=h264_vaapi,hevc_vaapi,av1_vaapi,vp9_vaapi)
        list(APPEND crossvmaf_decode_codec_sources libavcodec/vaapi_decode.c libavcodec/vaapi_h264.c
            libavcodec/vaapi_hevc.c libavcodec/vaapi_av1.c libavcodec/vaapi_vp9.c libavcodec/h265_profile_level.c)
        foreach(codec H264 HEVC AV1 VP9)
            list(APPEND crossvmaf_decode_required_components CONFIG_${codec}_VAAPI_HWACCEL)
        endforeach()
    endif()
endif()

# A small pkg-config overlay for the two pinned dependencies. Unknown packages
# are delegated to the system pkg-config; Linux libva is never synthesized.
find_program(crossvmaf_system_pkg_config NAMES pkg-config pkgconf)
set(crossvmaf_pkg_overlay "${CMAKE_BINARY_DIR}/_deps/decode-pkg")
file(MAKE_DIRECTORY "${crossvmaf_pkg_overlay}")
file(WRITE "${crossvmaf_pkg_overlay}/packages.json" "{\n")
if(CROSSVMAF_ENABLE_QSV)
    file(APPEND "${crossvmaf_pkg_overlay}/packages.json"
        "\"vpl\": {\"version\": \"2.15.0\", \"cflags\": \"-I${crossvmaf_vpl_install}/include -I${crossvmaf_vpl_install}/include/vpl\", \"libs\": \"-L${crossvmaf_vpl_install}/lib -lvpl\"},\n")
endif()
if(CROSSVMAF_ENABLE_NVDEC)
    file(APPEND "${crossvmaf_pkg_overlay}/packages.json"
        "\"ffnvcodec\": {\"version\": \"12.1.14.0\", \"cflags\": \"-I${crossvmaf_nvcodec_SOURCE_DIR}/include\", \"libs\": \"\"},\n")
endif()
file(APPEND "${crossvmaf_pkg_overlay}/packages.json" "\"_system\": \"${crossvmaf_system_pkg_config}\"\n}\n")
set(crossvmaf_pkg_python "${Python3_EXECUTABLE}")
if(WIN32)
    string(REGEX REPLACE "^([A-Za-z]):/" "/\\1/" crossvmaf_pkg_python "${crossvmaf_pkg_python}")
endif()
file(WRITE "${crossvmaf_pkg_overlay}/pkg-config" "#!/bin/sh\nexec '${crossvmaf_pkg_python}' '${CMAKE_CURRENT_LIST_DIR}/ffmpeg/PkgConfig.py' '${crossvmaf_pkg_overlay}/packages.json' \"$@\"\n")
if(NOT WIN32)
    file(CHMOD "${crossvmaf_pkg_overlay}/pkg-config" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()
if(CROSSVMAF_ENABLE_QSV OR CROSSVMAF_ENABLE_NVDEC)
    list(APPEND crossvmaf_decode_configure_arguments "--pkg-config=${crossvmaf_pkg_overlay}/pkg-config")
endif()
string(JOIN "\n" crossvmaf_decode_arguments_text ${crossvmaf_decode_configure_arguments})
file(WRITE "${crossvmaf_pkg_overlay}/configure-arguments.txt" "${crossvmaf_decode_arguments_text}\n")
file(SHA256 "${CMAKE_CURRENT_LIST_DIR}/ffmpeg/PkgConfig.py" crossvmaf_pkg_script_identity)
file(SHA256 "${crossvmaf_pkg_overlay}/packages.json" crossvmaf_pkg_packages_identity)
string(SHA256 crossvmaf_decode_configuration_identity "${crossvmaf_decode_arguments_text};${CROSSVMAF_ENABLE_VULKAN_VIDEO};${CROSSVMAF_ENABLE_QSV};${CROSSVMAF_ENABLE_NVDEC};${CROSSVMAF_ENABLE_VAAPI};${crossvmaf_pkg_script_identity};${crossvmaf_pkg_packages_identity}")
