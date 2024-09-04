add_library(separate STATIC separate_engine.cc)
target_link_libraries(separate PUBLIC frontend ${TORCH_LIBRARIES})
