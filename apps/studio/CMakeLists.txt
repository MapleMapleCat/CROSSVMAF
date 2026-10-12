# UI translations, one gettext catalog per language in the layout Slint's
# bundler expects: i18n/<lang>/LC_MESSAGES/crossvmaf_studio.po. The same files
# feed Slint's @tr strings and the C++ i18n::tr tables. docs/STUDIO_I18N.md.
set(CROSSVMAF_STUDIO_I18N_DIR "${CMAKE_CURRENT_SOURCE_DIR}/i18n")
file(GLOB crossvmaf_studio_catalogs CONFIGURE_DEPENDS
    "${CROSSVMAF_STUDIO_I18N_DIR}/*/LC_MESSAGES/crossvmaf_studio.po")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(crossvmaf_studio_catalog_tables
    "${CMAKE_CURRENT_BINARY_DIR}/generated/i18n/studio_catalog.inc")
add_custom_command(
    OUTPUT "${crossvmaf_studio_catalog_tables}"
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/i18n/studio_i18n.py"
            generate-cpp --output "${crossvmaf_studio_catalog_tables}"
            ${crossvmaf_studio_catalogs}
    DEPENDS
        "${PROJECT_SOURCE_DIR}/tools/i18n/studio_i18n.py"
        ${crossvmaf_studio_catalogs}
    VERBATIM
    COMMENT "Generating Studio translation tables"
)

set(crossvmaf_pdf_fonts "${CMAKE_CURRENT_BINARY_DIR}/generated/pdf_fonts.cpp")
set(crossvmaf_pdf_font_files
    "${CMAKE_CURRENT_SOURCE_DIR}/ui/fonts/Inter-Regular.ttf"
    "${CMAKE_CURRENT_SOURCE_DIR}/ui/fonts/Inter-SemiBold.ttf"
    "${CMAKE_CURRENT_SOURCE_DIR}/ui/fonts/NotoSansSC-Regular.ttf")
add_custom_command(OUTPUT "${crossvmaf_pdf_fonts}"
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/studio/embed_pdf_fonts.py"
        --output "${crossvmaf_pdf_fonts}" ${crossvmaf_pdf_font_files}
    DEPENDS "${PROJECT_SOURCE_DIR}/tools/studio/embed_pdf_fonts.py" ${crossvmaf_pdf_font_files}
    VERBATIM COMMENT "Embedding portable PDF fonts")

# Application logic has no Slint/generated-UI dependency and is testable
# without creating a window or running the UI event loop.
add_library(crossvmaf_studio_support STATIC
    src/types.cpp
    src/types.hpp
    src/i18n/catalog.hpp
    src/i18n/language.cpp
    src/i18n/language.hpp
    src/i18n/translator.cpp
    src/i18n/translator.hpp
    "${crossvmaf_studio_catalog_tables}"
    src/app/settings.cpp
    src/app/settings.hpp
    src/app/settings_store.cpp
    src/app/settings_store.hpp
    src/evaluation/comparison_setup.cpp
    src/evaluation/comparison_setup.hpp
    src/evaluation/probe_service.cpp
    src/evaluation/probe_service.hpp
    src/evaluation/scene_policy.cpp
    src/evaluation/scene_policy.hpp
    src/evaluation/task_queue.cpp
    src/evaluation/task_queue.hpp
    src/evaluation/evaluation_request.hpp
    src/evaluation/evaluation_progress.hpp
    src/evaluation/evaluation_runner.cpp
    src/evaluation/evaluation_runner.hpp
    src/reports/evaluation_report.hpp
    src/reports/report_analysis.cpp
    src/reports/report_analysis.hpp
    src/reports/report_writer.cpp
    src/reports/report_writer.hpp
    src/reports/report_pdf.cpp
    src/reports/report_pdf.hpp
    src/reports/pdf_document.cpp
    src/reports/pdf_document.hpp
    "${crossvmaf_pdf_fonts}"
    src/reports/comparison.cpp
    src/reports/comparison.hpp
    src/reports/comparison_selection.cpp
    src/reports/comparison_selection.hpp
    src/reports/comparison_session.cpp
    src/reports/comparison_session.hpp
    src/reports/report_dto.cpp
    src/reports/report_dto.hpp
    src/reports/report_store.cpp
    src/reports/report_store.hpp
    src/platform/diagnostics.cpp
    src/platform/diagnostics.hpp
)
crossvmaf_configure_target(crossvmaf_studio_support)
target_include_directories(crossvmaf_studio_support
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(crossvmaf_studio_support
    PUBLIC CrossVMAF::crossvmaf CrossVMAF::version
    PRIVATE Threads::Threads nlohmann_json::nlohmann_json
)

add_executable(crossvmaf_studio
    src/application_main.cpp
    src/app/studio_controller.cpp
    src/app/studio_controller.hpp
    src/presentation/report_presenter.cpp
    src/presentation/report_presenter.hpp
    src/presentation/chart_data_builder.cpp
    src/presentation/chart_data_builder.hpp
    src/presentation/display_format.cpp
    src/presentation/display_format.hpp
    src/presentation/ui_language.cpp
    src/presentation/ui_language.hpp
    src/platform/native_dialogs.cpp
    src/platform/native_dialogs.hpp
    src/platform/window_integration.cpp
    src/platform/window_integration.hpp
)
set_target_properties(crossvmaf_studio PROPERTIES
    OUTPUT_NAME crossvmaf-studio
    # Canonical artifacts must remain isolated by build tree and configuration.
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
)
target_compile_features(crossvmaf_studio PRIVATE cxx_std_20)
if(MSVC)
    # The Slint headers and the generated .slint C++ code are not /W4 clean;
    # keep the project's conformance flags off this target.
    target_compile_options(crossvmaf_studio PRIVATE
        /permissive- /Zc:__cplusplus /utf-8 /W3 /wd4244 /wd4201 /wd4458 /wd4060 /wd4624)
endif()
target_include_directories(crossvmaf_studio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(crossvmaf_studio PRIVATE
    crossvmaf_studio_support
    nlohmann_json::nlohmann_json
    Slint::Slint
)
# Embed UI assets so distributed builds do not load icons or fonts from the
# build machine's source paths. Also bundle catalogs: @tr strings carry no
# implicit component context, so one msgid is translated once for all components.
set_target_properties(crossvmaf_studio PROPERTIES
    SLINT_EMBED_RESOURCES "embed-files"
    SLINT_BUNDLE_TRANSLATIONS "${CROSSVMAF_STUDIO_I18N_DIR}"
    SLINT_NO_DEFAULT_TRANSLATION_CONTEXT ON)
slint_target_sources(crossvmaf_studio ui/app.slint
    LIBRARY_PATHS "crossvmaf-version=${CROSSVMAF_VERSION_SLINT}")
# slint-compiler's depfile lists .slint imports but not the catalogs.
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/app.h" APPEND
    DEPENDS "${CROSSVMAF_VERSION_SLINT}" ${crossvmaf_studio_catalogs})
crossvmaf_stage_studio_licenses(crossvmaf_studio)
if(WIN32)
    target_link_libraries(crossvmaf_studio PRIVATE comctl32 comdlg32 shell32)
    if(TARGET slint_cpp-shared)
        add_custom_command(TARGET crossvmaf_studio POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:slint_cpp-shared>" "$<TARGET_FILE_DIR:crossvmaf_studio>"
            VERBATIM)
    else()
        # Native dependencies of Slint's winit/FemtoVG static runtime are not
        # propagated by the upstream Rust/CMake integration.
        target_link_libraries(crossvmaf_studio PRIVATE imm32 opengl32)
    endif()
endif()
