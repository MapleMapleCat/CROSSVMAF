add_library(crossvmaf_numeric STATIC
    q32.cpp
    decimal.cpp
    transcendentals.cpp
)
crossvmaf_use_project_headers(crossvmaf_numeric)
crossvmaf_configure_target(crossvmaf_numeric)
