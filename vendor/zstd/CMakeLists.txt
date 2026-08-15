file(GLOB ZSTD_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compress/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/decompress/*.c
)
add_library(zstd STATIC ${ZSTD_SRC})
target_include_directories(zstd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(zstd PRIVATE XXH_NAMESPACE=ZSTD_ ZSTD_LEGACY_SUPPORT=0)
set_target_properties(zstd PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_definitions(zstd PRIVATE ZSTD_DISABLE_ASM)
else()
    enable_language(ASM)
    target_sources(zstd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/decompress/huf_decompress_amd64.S)
endif()
