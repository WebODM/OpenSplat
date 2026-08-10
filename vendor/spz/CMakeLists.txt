find_package(zstd QUIET)
if(zstd_FOUND)
    if(TARGET zstd::libzstd_static)
        set(SPZ_ZSTD_LIB zstd::libzstd_static)
    else()
        set(SPZ_ZSTD_LIB zstd::libzstd_shared)
    endif()
else()
    include(FetchContent)
    FetchContent_Declare(zstd
        URL https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz
        URL_HASH SHA256=8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1
        SOURCE_SUBDIR build/cmake
    )
    set(ZSTD_BUILD_SHARED OFF)
    set(ZSTD_BUILD_PROGRAMS OFF)
    set(ZSTD_BUILD_TESTS OFF)
    set(ZSTD_LEGACY_SUPPORT OFF)
    FetchContent_MakeAvailable(zstd)
    set(SPZ_ZSTD_LIB libzstd_static)
endif()

add_library(spz STATIC load-spz.cc splat-types.cc)
add_library(spz::spz ALIAS spz)
set_target_properties(spz PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(spz PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spz PRIVATE ${ZLIB_LIB} ${SPZ_ZSTD_LIB})
