add_library(miniz STATIC miniz.c)
target_include_directories(miniz PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(miniz PROPERTIES POSITION_INDEPENDENT_CODE ON)
