add_library(crossvmaf_scoring STATIC
    model.cpp
    predictor.cpp
    pooling.cpp
)
crossvmaf_use_project_headers(crossvmaf_scoring)
crossvmaf_configure_target(crossvmaf_scoring)
target_link_libraries(crossvmaf_scoring PUBLIC crossvmaf_numeric)
crossvmaf_generate_model_pack(crossvmaf_scoring)
