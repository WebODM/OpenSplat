add_library(spz STATIC load-spz.cc splat-types.cc)
add_library(spz::spz ALIAS spz)
set_target_properties(spz PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(spz PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spz PRIVATE miniz zstd)
