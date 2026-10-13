set(CROSSVMAF_FFMPEG_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/third_party/ffmpeg")
set(CROSSVMAF_FFMPEG_ABI_MAJOR 1)
if(NOT EXISTS "${CROSSVMAF_FFMPEG_SOURCE_DIR}/configure" OR
   NOT EXISTS "${CROSSVMAF_FFMPEG_SOURCE_DIR}/VERSION")
    message(FATAL_ERROR
        "Complete FFmpeg sources are missing from third_party/ffmpeg")
endif()

file(READ "${CROSSVMAF_FFMPEG_SOURCE_DIR}/VERSION"
    crossvmaf_ffmpeg_vendored_version)
string(STRIP "${crossvmaf_ffmpeg_vendored_version}"
    crossvmaf_ffmpeg_vendored_version)
if(NOT crossvmaf_ffmpeg_vendored_version STREQUAL "9.0.1")
    message(FATAL_ERROR
        "third_party/ffmpeg must contain the complete FFmpeg 9.0.1 source tree")
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" crossvmaf_system_processor)
if(NOT crossvmaf_system_processor MATCHES "^(amd64|x86_64)$")
    message(FATAL_ERROR
        "The initial bundled FFmpeg source target supports x86-64 only")
endif()

set(crossvmaf_ffmpeg_configuration_directory
    "${CMAKE_BINARY_DIR}/_deps/crossvmaf-ffmpeg-config")
set(crossvmaf_ffmpeg_configuration_identity
    "ffmpeg-9.0.1-${CMAKE_SYSTEM_NAME}-${CMAKE_C_COMPILER}-${CMAKE_C_COMPILER_VERSION}-lgpl-vulkan-video-dav1d-x86asm-v16-${crossvmaf_decode_configuration_identity}")
set(crossvmaf_ffmpeg_configuration_stamp
    "${crossvmaf_ffmpeg_configuration_directory}/crossvmaf-configure-stamp.txt")

set(crossvmaf_ffmpeg_configure_arguments
    "--cc=${CMAKE_C_COMPILER}"
    "--cxx=${CMAKE_CXX_COMPILER}"
    --disable-programs
    --enable-shared
    --disable-static
    --disable-symver
    --disable-doc
    --disable-debug
    --disable-gpl
    --disable-version3
    --disable-nonfree
    --disable-autodetect
    --disable-libaom
    --disable-network
    --disable-avdevice
    --disable-avfilter
    --disable-swscale
    --disable-swresample
    --disable-encoders
    --disable-muxers
    --disable-filters
    --disable-devices
    --disable-everything
    --enable-vulkan
    --enable-avcodec
    --enable-avformat
    --enable-avutil
    --enable-decoder=h264,hevc,av1,vp8,vp9,mpeg1video,mpeg2video,mpeg4,vc1,wmv3,flv,mjpeg
    ${crossvmaf_decode_configure_arguments}
    --enable-demuxer=h264,hevc,av1,obu,ivf,m4v,mpegvideo,mpegps,mpegts,vc1,mjpeg,mpjpeg,mov,matroska,avi,asf,flv,dv
    --enable-parser=h264,hevc,av1,vp8,vp9,mpegvideo,mpeg4video,vc1,mjpeg
    --enable-protocol=file
    --disable-inline-asm
    "--x86asmexe=${CROSSVMAF_NASM_EXECUTABLE}"
    "--extra-cflags=-I${vulkanheaders_SOURCE_DIR}/include"
)
if(NOT MSVC)
    # CMake links this closure as a shared library, including the NASM objects.
    list(APPEND crossvmaf_ffmpeg_configure_arguments --enable-pic)
endif()
string(SHA256 crossvmaf_ffmpeg_arguments_identity "${crossvmaf_ffmpeg_configure_arguments}")
string(APPEND crossvmaf_ffmpeg_configuration_identity "-${crossvmaf_ffmpeg_arguments_identity}")

if(MSVC)
    find_program(crossvmaf_ffmpeg_git_bash
        NAMES bash.exe
        HINTS
            "$ENV{ProgramFiles}/Git/bin"
            "$ENV{ProgramW6432}/Git/bin"
        NO_DEFAULT_PATH
        REQUIRED
    )
    find_program(crossvmaf_ffmpeg_powershell
        NAMES pwsh.exe powershell.exe
        REQUIRED
    )
    set(crossvmaf_ffmpeg_environment_arguments)
    if(CMAKE_GENERATOR_INSTANCE)
        list(APPEND crossvmaf_ffmpeg_environment_arguments
            -VisualStudioEnvironmentScript
            "${CMAKE_GENERATOR_INSTANCE}/VC/Auxiliary/Build/vcvars64.bat")
    endif()

    execute_process(
        COMMAND
            "${crossvmaf_ffmpeg_powershell}"
            -NoProfile
            -ExecutionPolicy Bypass
            -File
                "${CMAKE_CURRENT_LIST_DIR}/ffmpeg/windows/ConfigureMsvc.ps1"
            -SourceDirectory "${CROSSVMAF_FFMPEG_SOURCE_DIR}"
            -ConfigurationDirectory
                "${crossvmaf_ffmpeg_configuration_directory}"
            -GitBashExecutable "${crossvmaf_ffmpeg_git_bash}"
            -CompilerExecutable "${CMAKE_C_COMPILER}"
            -CompilerVersion "${CMAKE_C_COMPILER_VERSION}"
            ${crossvmaf_ffmpeg_environment_arguments}
            -VulkanHeadersDirectory "${vulkanheaders_SOURCE_DIR}/include"
            -NasmExecutable "${CROSSVMAF_NASM_EXECUTABLE}"
            -BackendArgumentsFile "${crossvmaf_pkg_overlay}/configure-arguments.txt"
            -BackendConfigurationIdentity "${crossvmaf_decode_configuration_identity}"
        RESULT_VARIABLE crossvmaf_ffmpeg_configure_result
        COMMAND_ECHO STDOUT
    )
else()
    set(crossvmaf_ffmpeg_requires_configuration TRUE)
    if(EXISTS "${crossvmaf_ffmpeg_configuration_stamp}")
        file(READ "${crossvmaf_ffmpeg_configuration_stamp}"
            crossvmaf_installed_ffmpeg_configuration_identity)
        if(crossvmaf_installed_ffmpeg_configuration_identity STREQUAL
           crossvmaf_ffmpeg_configuration_identity AND
           EXISTS "${crossvmaf_ffmpeg_configuration_directory}/config.h" AND
           EXISTS "${crossvmaf_ffmpeg_configuration_directory}/config.asm")
            set(crossvmaf_ffmpeg_requires_configuration FALSE)
        endif()
    endif()

    if(crossvmaf_ffmpeg_requires_configuration)
        file(REMOVE_RECURSE "${crossvmaf_ffmpeg_configuration_directory}")
        file(MAKE_DIRECTORY "${crossvmaf_ffmpeg_configuration_directory}")
        execute_process(
            COMMAND
                "${CROSSVMAF_FFMPEG_SOURCE_DIR}/configure"
                ${crossvmaf_ffmpeg_configure_arguments}
            WORKING_DIRECTORY "${crossvmaf_ffmpeg_configuration_directory}"
            RESULT_VARIABLE crossvmaf_ffmpeg_configure_result
            COMMAND_ECHO STDOUT
        )
        if(crossvmaf_ffmpeg_configure_result EQUAL 0)
            file(WRITE "${crossvmaf_ffmpeg_configuration_stamp}"
                "${crossvmaf_ffmpeg_configuration_identity}")
        endif()
    else()
        set(crossvmaf_ffmpeg_configure_result 0)
    endif()
endif()

if(NOT crossvmaf_ffmpeg_configure_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to configure the bundled FFmpeg 9.0.1 source target")
endif()

file(READ "${crossvmaf_ffmpeg_configuration_directory}/config.h"
    crossvmaf_ffmpeg_generated_configuration)
if(NOT crossvmaf_ffmpeg_generated_configuration MATCHES "#define CONFIG_SHARED 1" OR
   NOT crossvmaf_ffmpeg_generated_configuration MATCHES "#define CONFIG_STATIC 0" OR
   NOT crossvmaf_ffmpeg_generated_configuration MATCHES "#define HAVE_SYMVER 0")
    message(FATAL_ERROR "FFmpeg must be configured for the explicitly exported combined shared library")
endif()
if(NOT crossvmaf_ffmpeg_generated_configuration MATCHES
   "#define FFMPEG_LICENSE \"LGPL version 2\\.1 or later\"")
    message(FATAL_ERROR
        "Bundled FFmpeg configuration is not the required LGPL-only build")
endif()
if(NOT crossvmaf_ffmpeg_generated_configuration MATCHES
   "#define CONFIG_VULKAN 1")
    message(FATAL_ERROR
        "Bundled FFmpeg configuration is missing Vulkan support")
endif()
if(NOT crossvmaf_ffmpeg_generated_configuration MATCHES
   "#define HAVE_X86ASM 1")
    message(FATAL_ERROR
        "Bundled FFmpeg configuration is missing x86-64 assembly support")
endif()
file(READ
    "${crossvmaf_ffmpeg_configuration_directory}/config_components.h"
    crossvmaf_ffmpeg_generated_components)
set(crossvmaf_ffmpeg_required_components
    CONFIG_H264_DECODER
    CONFIG_HEVC_DECODER
    CONFIG_AV1_DECODER
    CONFIG_VP8_DECODER
    CONFIG_VP9_DECODER
    CONFIG_MPEG1VIDEO_DECODER
    CONFIG_MPEG2VIDEO_DECODER
    CONFIG_MPEG4_DECODER
    CONFIG_VC1_DECODER
    CONFIG_WMV3_DECODER
    CONFIG_FLV_DECODER
    CONFIG_MJPEG_DECODER
    CONFIG_H264_DEMUXER
    CONFIG_HEVC_DEMUXER
    CONFIG_AV1_DEMUXER
    CONFIG_OBU_DEMUXER
    CONFIG_IVF_DEMUXER
    CONFIG_M4V_DEMUXER
    CONFIG_MPEGVIDEO_DEMUXER
    CONFIG_MPEGPS_DEMUXER
    CONFIG_MPEGTS_DEMUXER
    CONFIG_VC1_DEMUXER
    CONFIG_MJPEG_DEMUXER
    CONFIG_MPJPEG_DEMUXER
    CONFIG_MOV_DEMUXER
    CONFIG_MATROSKA_DEMUXER
    CONFIG_AVI_DEMUXER
    CONFIG_ASF_DEMUXER
    CONFIG_FLV_DEMUXER
    CONFIG_DV_DEMUXER
    CONFIG_FILE_PROTOCOL
    ${crossvmaf_decode_required_components})
foreach(crossvmaf_ffmpeg_required_component IN LISTS
        crossvmaf_ffmpeg_required_components)
    if(NOT crossvmaf_ffmpeg_generated_components MATCHES
       "#define ${crossvmaf_ffmpeg_required_component} 1")
        message(FATAL_ERROR
            "Bundled FFmpeg configuration is missing required component: "
            "${crossvmaf_ffmpeg_required_component}")
    endif()
endforeach()

# FFmpeg configure cannot probe a CMake target that has not been built yet.
# Register FFmpeg's official libdav1d decoder wrapper explicitly while the
# parent build supplies its headers and static-library dependency below.
set(crossvmaf_ffmpeg_codec_registry
    "${crossvmaf_ffmpeg_configuration_directory}/libavcodec/codec_list.c")
if(NOT EXISTS "${crossvmaf_ffmpeg_codec_registry}")
    message(FATAL_ERROR
        "Bundled FFmpeg configuration did not generate its codec registry")
endif()
file(READ "${crossvmaf_ffmpeg_codec_registry}"
    crossvmaf_ffmpeg_codec_registry_contents)
if(NOT crossvmaf_ffmpeg_generated_configuration MATCHES
       "#define CONFIG_LIBAOM 0" OR
   NOT crossvmaf_ffmpeg_generated_components MATCHES
       "#define CONFIG_LIBAOM_AV1_DECODER 0" OR
   crossvmaf_ffmpeg_codec_registry_contents MATCHES "&ff_libaom_")
    message(FATAL_ERROR
        "Bundled FFmpeg must not enable libaom; AV1 software decoding requires dav1d")
endif()
if(NOT crossvmaf_ffmpeg_codec_registry_contents MATCHES
   "&ff_libdav1d_decoder,")
    if(NOT crossvmaf_ffmpeg_codec_registry_contents MATCHES
       "&ff_av1_decoder,")
        message(FATAL_ERROR
            "Bundled FFmpeg codec registry is missing the AV1 insertion point")
    endif()
    string(REPLACE
        "    &ff_av1_decoder,\n"
        "    &ff_libdav1d_decoder,\n    &ff_av1_decoder,\n"
        crossvmaf_ffmpeg_codec_registry_contents
        "${crossvmaf_ffmpeg_codec_registry_contents}")
    file(WRITE "${crossvmaf_ffmpeg_codec_registry}"
        "${crossvmaf_ffmpeg_codec_registry_contents}")
endif()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/third_party/licenses")
configure_file(
    "${CROSSVMAF_FFMPEG_SOURCE_DIR}/COPYING.LGPLv2.1"
    "${CMAKE_BINARY_DIR}/third_party/licenses/FFmpeg-LGPL-2.1.txt"
    COPYONLY
)
configure_file(
    "${dav1d_SOURCE_DIR}/COPYING"
    "${CMAKE_BINARY_DIR}/third_party/licenses/dav1d-BSD-2-Clause.txt"
    COPYONLY
)

file(MAKE_DIRECTORY
    "${crossvmaf_ffmpeg_configuration_directory}/libavutil")
file(WRITE
    "${crossvmaf_ffmpeg_configuration_directory}/libavutil/ffversion.h"
    "/* Generated for the pinned CrossVMAF source integration. */\n"
    "#ifndef AVUTIL_FFVERSION_H\n"
    "#define AVUTIL_FFVERSION_H\n"
    "#define FFMPEG_VERSION \"9.0.1\"\n"
    "#endif /* AVUTIL_FFVERSION_H */\n")

include("${CMAKE_CURRENT_LIST_DIR}/ffmpeg/Sources.cmake")
list(APPEND crossvmaf_ffmpeg_avutil_sources ${crossvmaf_decode_util_sources})
list(APPEND crossvmaf_ffmpeg_avcodec_sources ${crossvmaf_decode_codec_sources})
if(NOT CROSSVMAF_ENABLE_VULKAN_VIDEO)
    list(FILTER crossvmaf_ffmpeg_avcodec_sources EXCLUDE REGEX "libavcodec/vulkan.*\\.c$")
endif()
list(TRANSFORM crossvmaf_ffmpeg_avutil_sources
    PREPEND "${CROSSVMAF_FFMPEG_SOURCE_DIR}/")
list(TRANSFORM crossvmaf_ffmpeg_avcodec_sources
    PREPEND "${CROSSVMAF_FFMPEG_SOURCE_DIR}/")
list(TRANSFORM crossvmaf_ffmpeg_avformat_sources
    PREPEND "${CROSSVMAF_FFMPEG_SOURCE_DIR}/")

if(WIN32)
    set(CMAKE_ASM_NASM_OBJECT_FORMAT win64)
else()
    set(CMAKE_ASM_NASM_OBJECT_FORMAT elf64)
endif()
set(CMAKE_ASM_NASM_COMPILER "${CROSSVMAF_NASM_EXECUTABLE}")
enable_language(ASM_NASM)

function(crossvmaf_configure_ffmpeg_source_target target)
    set_target_properties(${target} PROPERTIES
        C_STANDARD 17
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS ON
    )
    target_include_directories(${target} SYSTEM
        PUBLIC
            "${crossvmaf_ffmpeg_configuration_directory}"
            "${CROSSVMAF_FFMPEG_SOURCE_DIR}"
    )
    target_include_directories(${target}
        PRIVATE
            "${CROSSVMAF_FFMPEG_SOURCE_DIR}/compat/stdbit"
            "${CROSSVMAF_FFMPEG_SOURCE_DIR}/libavutil/x86"
            "${CROSSVMAF_FFMPEG_SOURCE_DIR}/libavcodec/x86"
            "${CROSSVMAF_FFMPEG_SOURCE_DIR}/libavcodec/x86/hevc"
            "${CROSSVMAF_FFMPEG_SOURCE_DIR}/libavcodec/x86/h26x"
    )
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:ASM_NASM>:-P${crossvmaf_ffmpeg_configuration_directory}/config.asm>"
    )
    target_compile_definitions(${target} PRIVATE
        HAVE_AV_CONFIG_H
        _ISOC11_SOURCE
        _FILE_OFFSET_BITS=64
        _LARGEFILE_SOURCE
    )

    if(MSVC)
        target_include_directories(${target} PRIVATE
            "${CROSSVMAF_FFMPEG_SOURCE_DIR}/compat/atomics/win32")
        target_compile_definitions(${target} PRIVATE
            WIN32_LEAN_AND_MEAN
            _USE_MATH_DEFINES
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_WARNINGS
            _WIN32_WINNT=0x0600
        )
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:C,CXX>:/W0>"
            "$<$<COMPILE_LANG_AND_ID:C,MSVC>:/options:strict>"
            "$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>"
        )
    else()
        target_compile_definitions(${target} PRIVATE
            _POSIX_C_SOURCE=200112
            _XOPEN_SOURCE=600
        )
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:C,CXX>:-w>"
            "$<$<COMPILE_LANGUAGE:C,CXX>:-fPIC>"
        )
    endif()
endfunction()

# The configured closure ships as one shared library so applications link
# FFmpeg dynamically (the LGPL-2.1 shared-library mechanism): replacing
# the ABI-compatible shared library with a user-rebuilt library requires no
# application relink. All upstream sources compile once into an object library.
set(crossvmaf_ffmpeg_object_sources
    ${crossvmaf_ffmpeg_avutil_sources}
    ${crossvmaf_ffmpeg_avcodec_sources}
    ${crossvmaf_ffmpeg_avformat_sources})
list(REMOVE_DUPLICATES crossvmaf_ffmpeg_object_sources)

# CMake's NASM task ignores the per-item ObjectFileName that basename
# collision disambiguation assigns (aes.asm next to aes.c, ...), leaving
# objects.txt out of sync with the flat objects actually produced. Keep
# distinct C and assembly objects: colliding .asm files use custom commands with
# unique output paths that the NASM task never sees.
set(crossvmaf_ffmpeg_stems)
set(crossvmaf_ffmpeg_colliding_stems)
foreach(crossvmaf_ffmpeg_source IN LISTS crossvmaf_ffmpeg_object_sources)
    get_filename_component(crossvmaf_ffmpeg_stem
        "${crossvmaf_ffmpeg_source}" NAME_WE)
    if(crossvmaf_ffmpeg_stem IN_LIST crossvmaf_ffmpeg_stems)
        list(APPEND crossvmaf_ffmpeg_colliding_stems "${crossvmaf_ffmpeg_stem}")
    else()
        list(APPEND crossvmaf_ffmpeg_stems "${crossvmaf_ffmpeg_stem}")
    endif()
endforeach()

set(crossvmaf_ffmpeg_asm_objects)
set(crossvmaf_ffmpeg_filtered_sources)
foreach(crossvmaf_ffmpeg_source IN LISTS crossvmaf_ffmpeg_object_sources)
    get_filename_component(crossvmaf_ffmpeg_stem
        "${crossvmaf_ffmpeg_source}" NAME_WE)
    if(crossvmaf_ffmpeg_source MATCHES "\\.asm$" AND
       crossvmaf_ffmpeg_stem IN_LIST crossvmaf_ffmpeg_colliding_stems)
        file(RELATIVE_PATH crossvmaf_ffmpeg_asm_relative
            "${CROSSVMAF_FFMPEG_SOURCE_DIR}" "${crossvmaf_ffmpeg_source}")
        string(REPLACE "/" "_" crossvmaf_ffmpeg_asm_output
            "${crossvmaf_ffmpeg_asm_relative}")
        set(crossvmaf_ffmpeg_asm_object
            "${CMAKE_BINARY_DIR}/_deps/crossvmaf-ffmpeg-asm/$<CONFIG>/${crossvmaf_ffmpeg_asm_output}.obj")
        add_custom_command(
            OUTPUT "${crossvmaf_ffmpeg_asm_object}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${CMAKE_BINARY_DIR}/_deps/crossvmaf-ffmpeg-asm/$<CONFIG>"
            COMMAND "${CROSSVMAF_NASM_EXECUTABLE}"
                -f "${CMAKE_ASM_NASM_OBJECT_FORMAT}"
                "-I${CROSSVMAF_FFMPEG_SOURCE_DIR}/"
                "-I${CROSSVMAF_FFMPEG_SOURCE_DIR}/libavutil/x86/"
                "-I${CROSSVMAF_FFMPEG_SOURCE_DIR}/libavcodec/x86/"
                "-I${CROSSVMAF_FFMPEG_SOURCE_DIR}/libavcodec/x86/hevc/"
                "-I${CROSSVMAF_FFMPEG_SOURCE_DIR}/libavcodec/x86/h26x/"
                "-P${crossvmaf_ffmpeg_configuration_directory}/config.asm"
                -o "${crossvmaf_ffmpeg_asm_object}"
                "${crossvmaf_ffmpeg_source}"
            DEPENDS
                "${crossvmaf_ffmpeg_source}"
                "${crossvmaf_ffmpeg_configuration_directory}/config.asm"
            VERBATIM
            COMMENT "Assembling ${crossvmaf_ffmpeg_asm_relative}"
        )
        list(APPEND crossvmaf_ffmpeg_asm_objects "${crossvmaf_ffmpeg_asm_object}")
    else()
        list(APPEND crossvmaf_ffmpeg_filtered_sources "${crossvmaf_ffmpeg_source}")
    endif()
endforeach()
add_custom_target(crossvmaf_ffmpeg_asm DEPENDS ${crossvmaf_ffmpeg_asm_objects})

add_library(crossvmaf_ffmpeg_objects OBJECT
    ${crossvmaf_ffmpeg_filtered_sources})
crossvmaf_configure_ffmpeg_source_target(crossvmaf_ffmpeg_objects)
if(CROSSVMAF_ENABLE_QSV)
    target_include_directories(crossvmaf_ffmpeg_objects SYSTEM PRIVATE "${crossvmaf_vpl_install}/include" "${crossvmaf_vpl_install}/include/vpl")
    target_compile_definitions(crossvmaf_ffmpeg_objects PRIVATE MFX_DEPRECATED_OFF)
endif()
if(CROSSVMAF_ENABLE_NVDEC)
    target_include_directories(crossvmaf_ffmpeg_objects SYSTEM PRIVATE "${crossvmaf_nvcodec_SOURCE_DIR}/include")
endif()
if(CROSSVMAF_ENABLE_AMF)
    target_include_directories(crossvmaf_ffmpeg_objects SYSTEM PRIVATE "${crossvmaf_amf_include}")
endif()
if(TARGET PkgConfig::CROSSVMAF_VA)
    target_link_libraries(crossvmaf_ffmpeg_objects PRIVATE PkgConfig::CROSSVMAF_VA)
endif()
set_target_properties(crossvmaf_ffmpeg_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(crossvmaf_ffmpeg_objects PUBLIC
    Vulkan::Headers
    dav1d::dav1d)
set_source_files_properties(${crossvmaf_ffmpeg_asm_objects}
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
add_library(crossvmaf_ffmpeg SHARED
    $<TARGET_OBJECTS:crossvmaf_ffmpeg_objects>
    ${crossvmaf_ffmpeg_asm_objects})
add_dependencies(crossvmaf_ffmpeg crossvmaf_ffmpeg_asm)
set_target_properties(crossvmaf_ffmpeg PROPERTIES
    OUTPUT_NAME crossvmaf-ffmpeg
    LINKER_LANGUAGE C
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

# The reviewed manifest is the only export authority. Inspection validates the
# linked artifact; it never discovers or adds exports to this contract.
set(CROSSVMAF_FFMPEG_ABI_CONTRACT "${CMAKE_CURRENT_LIST_DIR}/ffmpeg/abi/exports.txt")
set(CROSSVMAF_FFMPEG_ABI_TOOL "${CMAKE_CURRENT_LIST_DIR}/ffmpeg/abi/abi.py")
get_filename_component(crossvmaf_compiler_directory "${CMAKE_C_COMPILER}" DIRECTORY)
find_program(CROSSVMAF_LLVM_READOBJ NAMES llvm-readobj
    HINTS "${crossvmaf_compiler_directory}" REQUIRED)
set(crossvmaf_ffmpeg_abi_directory "${CMAKE_CURRENT_BINARY_DIR}/generated/ffmpeg-abi")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CROSSVMAF_FFMPEG_ABI_CONTRACT}" "${CROSSVMAF_FFMPEG_ABI_TOOL}")
execute_process(COMMAND "${Python3_EXECUTABLE}" -B "${CROSSVMAF_FFMPEG_ABI_TOOL}"
    generate --contract "${CROSSVMAF_FFMPEG_ABI_CONTRACT}"
    --output "${crossvmaf_ffmpeg_abi_directory}"
    --library "crossvmaf-ffmpeg"
    COMMAND_ERROR_IS_FATAL ANY)
if(WIN32)
    target_sources(crossvmaf_ffmpeg PRIVATE "${crossvmaf_ffmpeg_abi_directory}/exports.def")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(crossvmaf_ffmpeg PROPERTIES
        VERSION "${CROSSVMAF_FFMPEG_ABI_MAJOR}.0.0" SOVERSION "${CROSSVMAF_FFMPEG_ABI_MAJOR}")
else()
    message(FATAL_ERROR "The FFmpeg shared ABI currently supports Windows and Linux")
endif()
add_custom_command(TARGET crossvmaf_ffmpeg POST_BUILD
    COMMAND "${Python3_EXECUTABLE}" -B "${CROSSVMAF_FFMPEG_ABI_TOOL}"
        check-exports --contract "${CROSSVMAF_FFMPEG_ABI_CONTRACT}"
        --reader "${CROSSVMAF_LLVM_READOBJ}" --binary "$<TARGET_FILE:crossvmaf_ffmpeg>"
    VERBATIM)
# Also audit on no-op builds, including after validator changes or replacement
# of an existing DLL. Consumers cannot finish a build with an unchecked ABI.
add_custom_target(crossvmaf_ffmpeg_abi_check ALL
    COMMAND "${Python3_EXECUTABLE}" -B "${CROSSVMAF_FFMPEG_ABI_TOOL}"
        check-exports --contract "${CROSSVMAF_FFMPEG_ABI_CONTRACT}"
        --reader "${CROSSVMAF_LLVM_READOBJ}" --binary "$<TARGET_FILE:crossvmaf_ffmpeg>"
    DEPENDS crossvmaf_ffmpeg VERBATIM)
target_include_directories(crossvmaf_ffmpeg SYSTEM PUBLIC
    "${crossvmaf_ffmpeg_configuration_directory}"
    "${CROSSVMAF_FFMPEG_SOURCE_DIR}")
target_link_libraries(crossvmaf_ffmpeg PUBLIC Vulkan::Headers)
target_link_libraries(crossvmaf_ffmpeg PRIVATE
    dav1d::dav1d
    Threads::Threads)
if(WIN32)
    target_link_libraries(crossvmaf_ffmpeg PRIVATE bcrypt ole32 user32)
else()
    target_link_libraries(crossvmaf_ffmpeg PRIVATE ${CMAKE_DL_LIBS} m)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Match upstream's symbol boundary. Internal ff_* assembly references
        # must bind locally rather than become preemptible ELF exports.
        target_link_options(crossvmaf_ffmpeg PRIVATE
            "LINKER:--no-undefined"
            "LINKER:--version-script=${crossvmaf_ffmpeg_abi_directory}/exports.map")
        set_property(TARGET crossvmaf_ffmpeg APPEND PROPERTY LINK_DEPENDS
            "${crossvmaf_ffmpeg_abi_directory}/exports.map")
        set_property(TARGET crossvmaf_ffmpeg PROPERTY BUILD_RPATH "$ORIGIN")
    endif()
endif()

add_library(FFmpeg::avutil ALIAS crossvmaf_ffmpeg)
if(CROSSVMAF_ENABLE_QSV)
    target_link_libraries(crossvmaf_ffmpeg PRIVATE CrossVMAF::vpl)
endif()
if(TARGET PkgConfig::CROSSVMAF_VA)
    target_link_libraries(crossvmaf_ffmpeg PRIVATE PkgConfig::CROSSVMAF_VA)
endif()
add_library(FFmpeg::avcodec ALIAS crossvmaf_ffmpeg)
add_library(FFmpeg::avformat ALIAS crossvmaf_ffmpeg)

function(crossvmaf_stage_ffmpeg_runtime target)
    add_dependencies(${target} crossvmaf_ffmpeg_abi_check)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "$ORIGIN")
        set_property(TARGET ${target} PROPERTY INSTALL_RPATH "$ORIGIN")
    endif()
    if(CROSSVMAF_ENABLE_QSV)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:CrossVMAF::vpl>" "$<TARGET_FILE_DIR:${target}>" VERBATIM)
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:crossvmaf_ffmpeg>"
            "$<TARGET_FILE_DIR:${target}>"
        VERBATIM
        COMMENT "Staging the FFmpeg shared library beside ${target}"
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:crossvmaf_ffmpeg>"
                "$<TARGET_FILE_DIR:${target}>/$<TARGET_SONAME_FILE_NAME:crossvmaf_ffmpeg>"
            VERBATIM)
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${Python3_EXECUTABLE}" -B "${CROSSVMAF_FFMPEG_ABI_TOOL}"
            check-imports --contract "${CROSSVMAF_FFMPEG_ABI_CONTRACT}"
            --reader "${CROSSVMAF_LLVM_READOBJ}" --binary "$<TARGET_FILE:${target}>"
            --library "$<TARGET_FILE_NAME:crossvmaf_ffmpeg>" VERBATIM)
endfunction()

function(crossvmaf_stage_codec_licenses target)
    crossvmaf_stage_ffmpeg_runtime(${target})
    crossvmaf_stage_license(${target}
        "${CMAKE_BINARY_DIR}/third_party/licenses/FFmpeg-LGPL-2.1.txt" "FFmpeg-LGPL-2.1.txt")
    crossvmaf_stage_license(${target}
        "${PROJECT_SOURCE_DIR}/third_party/licenses/FFmpeg-9.0.1-NOTICE.md" "FFmpeg-9.0.1-NOTICE.md")
    crossvmaf_stage_license(${target}
        "${CMAKE_BINARY_DIR}/third_party/licenses/dav1d-BSD-2-Clause.txt" "dav1d-BSD-2-Clause.txt")
    crossvmaf_stage_license(${target}
        "${PROJECT_SOURCE_DIR}/third_party/licenses/dav1d-1.5.4-NOTICE.md" "dav1d-1.5.4-NOTICE.md")
endfunction()
