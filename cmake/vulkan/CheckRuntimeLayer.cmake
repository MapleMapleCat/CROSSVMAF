if(NOT DEFINED RUNTIME_DIRECTORY OR NOT IS_DIRECTORY "${RUNTIME_DIRECTORY}")
    message(FATAL_ERROR
        "CheckRuntimeLayer.cmake requires RUNTIME_DIRECTORY")
endif()

file(GLOB_RECURSE runtime_source_files
    "${RUNTIME_DIRECTORY}/*.cpp"
    "${RUNTIME_DIRECTORY}/*.hpp")

foreach(runtime_source_file IN LISTS runtime_source_files)
    file(READ "${runtime_source_file}" runtime_source_contents)
    if(runtime_source_contents MATCHES
        "#[ \t]*include[ \t]*[\"<](video/|evaluation/|metrics/|scoring/|providers/|input/|libav)")
        message(FATAL_ERROR
            "GPU primitive includes a forbidden upper domain: "
            "${runtime_source_file}: ${CMAKE_MATCH_0}")
    endif()

    if(runtime_source_contents MATCHES
        "(video|evaluation|metrics|scoring|providers|input)::")
        message(FATAL_ERROR
            "GPU primitive refers to an upper domain namespace: "
            "${runtime_source_file}: ${CMAKE_MATCH_0}")
    endif()

    if(runtime_source_contents MATCHES
        "#[ \t]*include[ \t]*\"shaders/embedded_spirv_manifest.hpp\"")
        message(FATAL_ERROR
            "Vulkan runtime must receive the provider shader manifest identity "
            "through composition: ${runtime_source_file}")
    endif()

    if(runtime_source_contents MATCHES
        "(^|[^A-Za-z0-9_])(Cambi|CAMBI|cambi|SpeedChroma|speed_chroma|Adm3|ADM3|adm3|Motion3|motion3)([^A-Za-z0-9_]|$)")
        message(FATAL_ERROR
            "Vulkan runtime source encodes extractor-specific knowledge: "
            "${runtime_source_file}")
    endif()
endforeach()
