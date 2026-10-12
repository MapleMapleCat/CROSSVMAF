function(crossvmaf_configure_target target)
    target_compile_features(${target} PUBLIC cxx_std_20)

    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /Zc:__cplusplus /utf-8)
        if(CROSSVMAF_ENABLE_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
        )
        if(CROSSVMAF_ENABLE_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()

    # Vulkan aggregates initialize sType and intentionally zero every other
    # member. Apply this exception after /W4, which resets Clang's warning set.
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang,GNU>:-Wno-missing-field-initializers>")
endfunction()

function(crossvmaf_use_project_headers target)
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src
    )
endfunction()
