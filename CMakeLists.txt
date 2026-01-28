cmake_minimum_required(VERSION 3.21)
project(opensplat)

set(OPENSPLAT_BUILD_SIMPLE_TRAINER OFF CACHE BOOL "Build simple trainer applications")
set(GPU_RUNTIME "CUDA" CACHE STRING "HIP or CUDA or MPS")
set(OPENCV_DIR "OPENCV_DIR-NOTFOUND" CACHE PATH "Path to the OPENCV installation directory")
set(OPENSPLAT_MAX_CUDA_COMPATIBILITY OFF CACHE BOOL "Build for maximum CUDA device compatibility")
set(OPENSPLAT_BUILD_VISUALIZER OFF CACHE BOOL "Build visualizer application")
set(OPENSPLAT_USE_FAST_MATH OFF CACHE BOOL "Enable fast math optimizations for GPU kernels (-use_fast_math / -ffast-math)")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Read version
file(READ "VERSION" APP_VERSION)

# Read git commit
set(GIT_REV "")
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" 
                OUTPUT_VARIABLE GIT_REV 
                ERROR_QUIET)
string(REGEX REPLACE "\n$" "" GIT_REV "${GIT_REV}")
if (NOT "${GIT_REV}" STREQUAL "")
    set(DAPP_VERSION "${APP_VERSION} (git commit ${GIT_REV})")
    set(DAPP_REVISION "${GIT_REV}")
else()
    set(DAPP_VERSION "${APP_VERSION}")
    set(DAPP_REVISION "dev")
endif()

message("OpenSplat Version: ${DAPP_VERSION}")
add_compile_options("-DAPP_VERSION=\"${DAPP_VERSION}\"")
add_compile_options("-DAPP_REVISION=\"${DAPP_REVISION}\"")

# Don't complain about the override from NANOFLANN_BUILD_EXAMPLES
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# Use time-of-extraction for FetchContent'ed files modification time
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
# Suppress warning #20012-D (nvcc and glm)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -diag-suppress=20012)
if(OPENSPLAT_USE_FAST_MATH)
    message(STATUS "Fast math optimizations enabled for CUDA")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math)
endif()

include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
)
set(NANOFLANN_BUILD_EXAMPLES OFF)
set(NANOFLANN_BUILD_TESTS OFF)
FetchContent_Declare(nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.5.5.zip
)
FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.2.0.zip
)
FetchContent_MakeAvailable(nlohmann_json nanoflann cxxopts)
if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
    FetchContent_Declare(glm
        URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
    )
    FetchContent_MakeAvailable(glm)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if(GPU_RUNTIME STREQUAL "CUDA")
    find_package(CUDAToolkit)
    if (NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA toolkit not found, building with CPU support only")
        set(GPU_RUNTIME "CPU")
    else()
        if (OPENSPLAT_MAX_CUDA_COMPATIBILITY)
            execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-arch 
                    OUTPUT_VARIABLE LIST_GPU_ARCH 
                    ERROR_QUIET)
        endif()

        if(NOT LIST_GPU_ARCH AND OPENSPLAT_MAX_CUDA_COMPATIBILITY)
            message(WARNING "Cannot compile for max CUDA compatibility, nvcc does not support --list-gpu-arch")
            SET(OPENSPLAT_MAX_CUDA_COMPATIBILITY OFF)
        endif()
        if(NOT OPENSPLAT_MAX_CUDA_COMPATIBILITY)
            if(NOT CMAKE_CUDA_ARCHITECTURES)
                SET(CMAKE_CUDA_ARCHITECTURES 70;75;80)
            endif()
        else()
            # Build for maximum compatibility
            # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
            set(CMAKE_CUDA_ARCHITECTURES "")

            # Extract list of arch and gencodes
            string(REPLACE "\r" "" LIST_GPU_ARCH ${LIST_GPU_ARCH})
            string(REPLACE "\n" ";" LIST_GPU_ARCH ${LIST_GPU_ARCH})

            execute_process(COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-code 
                OUTPUT_VARIABLE LIST_GPU_CODE 
                ERROR_QUIET)
            string(REPLACE "\r" "" LIST_GPU_CODE ${LIST_GPU_CODE})
            string(REPLACE "\n" ";" LIST_GPU_CODE ${LIST_GPU_CODE})

            list(GET LIST_GPU_CODE 0 TARGET_GPU_CODE)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${TARGET_GPU_CODE}")
            
            set(IDX 0)
            foreach(GPU_ARCH ${LIST_GPU_ARCH})
                string(REGEX MATCH "compute_([0-9]+)" GPU_ARCH_VERSION "${GPU_ARCH}")
                list(APPEND CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}")
                list(GET LIST_GPU_CODE ${IDX} GPU_CODE)
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=${GPU_ARCH},code=${GPU_CODE}")
                math(EXPR IDX "${IDX}+1")
            endforeach()
            message("Set CUDA flags: " ${CMAKE_CUDA_FLAGS})
        endif()
        # Set torch cuda architecture list
        set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
        list(TRANSFORM TORCH_CUDA_ARCH_LIST REPLACE "^([0-9])([0-9])$" "\\1.\\2")
        list(TRANSFORM TORCH_CUDA_ARCH_LIST REPLACE "^([0-9][0-9])([0-9])$" "\\1.\\2")
        string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}")
        message(STATUS "** Updated TORCH_CUDA_ARCH_LIST to ${TORCH_CUDA_ARCH_LIST} **")
    endif()
elseif(GPU_RUNTIME STREQUAL "HIP")
    set(USE_HIP ON CACHE BOOL "Use HIP for GPU acceleration")

    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    find_package(HIP REQUIRED)
    if(OPENSPLAT_USE_FAST_MATH)
        message(STATUS "Fast math optimizations enabled for HIP")
        set(CMAKE_HIP_FLAGS ${CMAKE_HIP_FLAGS} -ffast-math)
    endif()

    file(GLOB_RECURSE GSPLAT_GPU_SRC LIST_DIRECTORIES False rasterizer/gsplat/*.cu)
    set_source_files_properties(${GSPLAT_GPU_SRC} PROPERTIES LANGUAGE HIP)

    if(WIN32)
        set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
    else()
        set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
elseif(GPU_RUNTIME STREQUAL "MPS")
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)
    message(STATUS "Metal framework found")

    set(XC_FLAGS -O3)
    if(OPENSPLAT_USE_FAST_MATH)
        message(STATUS "Fast math optimizations enabled for Metal")
        
        set(XC_FLAGS ${XC_FLAGS} -ffast-math)
    endif()
    set(USE_MPS ON CACHE BOOL "Use MPS for GPU acceleration")
else()
    set(GPU_RUNTIME "CPU")
endif()

set(CMAKE_CXX_STANDARD 17)
if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
    enable_language(${GPU_RUNTIME})
    set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
    set(${GPU_RUNTIME}_STANDARD 17)
endif()

if (NOT WIN32 AND NOT APPLE)
    set(STDPPFS_LIBRARY stdc++fs)
endif()

find_package(Torch REQUIRED)
find_package(OpenCV HINTS "${OPENCV_DIR}" REQUIRED)
if (OPENSPLAT_BUILD_VISUALIZER)
    find_package(Pangolin REQUIRED)
endif()

if (NOT WIN32 AND NOT APPLE)
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_calib3d)

set(GSPLAT_LIBS gsplat_cpu)
if((GPU_RUNTIME STREQUAL "CUDA") OR (GPU_RUNTIME STREQUAL "HIP"))
    add_library(gsplat rasterizer/gsplat/forward.cu rasterizer/gsplat/backward.cu rasterizer/gsplat/bindings.cu rasterizer/gsplat/helpers.cuh)
    list(APPEND GSPLAT_LIBS gsplat)
    if(GPU_RUNTIME STREQUAL "CUDA")
        set(GPU_LIBRARIES "cuda")
        target_link_libraries(gsplat PUBLIC cuda)
    else(GPU_RUNTIME STREQUAL "HIP")
        set(GPU_INCLUDE_DIRS "${ROCM_ROOT}/include")
        target_compile_definitions(gsplat PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    endif()
    target_include_directories(gsplat PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(gsplat PUBLIC glm::glm-header-only)
    set_target_properties(gsplat PROPERTIES LINKER_LANGUAGE CXX)
elseif(GPU_RUNTIME STREQUAL "MPS")    
    add_library(gsplat rasterizer/gsplat-metal/gsplat_metal.mm)
    list(APPEND GSPLAT_LIBS gsplat)
    target_link_libraries(gsplat PRIVATE 
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
    )
    target_include_directories(gsplat PRIVATE ${TORCH_INCLUDE_DIRS})
    # copy shader files to bin directory
    configure_file(rasterizer/gsplat-metal/gsplat_metal.metal ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.metal COPYONLY)
    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
        COMMAND xcrun -sdk macosx metal    ${XC_FLAGS} -c ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.metal -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.air
        COMMAND xcrun -sdk macosx metallib                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.air   -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
        COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.air
        COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gsplat_metal.metal
        DEPENDS rasterizer/gsplat-metal/gsplat_metal.metal
        COMMENT "Compiling Metal kernels"
    )

    add_custom_target(
        gsplat_metal ALL
        DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
    )
endif()

add_library(gsplat_cpu rasterizer/gsplat-cpu/gsplat_cpu.cpp)
target_include_directories(gsplat_cpu PRIVATE ${TORCH_INCLUDE_DIRS})

set(OPENSPLAT_SRC_FILES opensplat.cpp point_io.cpp nerfstudio.cpp model.cpp
kdtree_tensor.cpp spherical_harmonics.cpp cv_utils.cpp utils.cpp project_gaussians.cpp
rasterize_gaussians.cpp ssim.cpp optim_scheduler.cpp colmap.cpp opensfm.cpp openmvg.cpp input_data.cpp
tensor_math.cpp)

if (OPENSPLAT_BUILD_VISUALIZER)
    if (Pangolin_FOUND)
        message(STATUS "Found Pangolin. Building visualizer (beta)")
        list(APPEND OPENSPLAT_SRC_FILES visualizer.cpp)
        add_definitions(-DUSE_VISUALIZATION)
    else()
        message(FATAL "Pangolin not found. Cannot build visualizer (beta)")
    endif()
endif()

add_executable(opensplat ${OPENSPLAT_SRC_FILES})

install(TARGETS opensplat DESTINATION bin)
set_property(TARGET opensplat PROPERTY CXX_STANDARD 17)
target_include_directories(opensplat PRIVATE
    ${PROJECT_SOURCE_DIR}/rasterizer
    ${GPU_INCLUDE_DIRS}
)
target_link_libraries(opensplat PUBLIC ${STDPPFS_LIBRARY} ${GPU_LIBRARIES} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
if (Pangolin_FOUND)
    target_link_libraries(opensplat PUBLIC ${Pangolin_LIBRARIES})
endif()
target_link_libraries(opensplat PRIVATE
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    nanoflann::nanoflann
)
if (NOT WIN32)
    target_link_libraries(opensplat PUBLIC pthread)
endif()
if(GPU_RUNTIME STREQUAL "HIP")
    target_compile_definitions(opensplat PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
elseif(GPU_RUNTIME STREQUAL "CUDA")
    target_compile_definitions(opensplat PRIVATE USE_CUDA)
elseif(GPU_RUNTIME STREQUAL "MPS")
    target_compile_definitions(opensplat PRIVATE USE_MPS)
endif()

if(OPENSPLAT_BUILD_SIMPLE_TRAINER)
    add_executable(simple_trainer simple_trainer.cpp project_gaussians.cpp rasterize_gaussians.cpp cv_utils.cpp)
    install(TARGETS simple_trainer DESTINATION bin)
    target_include_directories(simple_trainer PRIVATE
        ${PROJECT_SOURCE_DIR}/rasterizer
        ${GPU_INCLUDE_DIRS}
    )
    target_link_libraries(simple_trainer PUBLIC ${GPU_LIBRARIES} ${GSPLAT_LIBS} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
    target_link_libraries(simple_trainer PRIVATE
        nlohmann_json::nlohmann_json
        cxxopts::cxxopts
        nanoflann::nanoflann
    )
    if (NOT WIN32)
        target_link_libraries(simple_trainer PUBLIC pthread)
    endif()
    set_property(TARGET simple_trainer PROPERTY CXX_STANDARD 17)
    if(GPU_RUNTIME STREQUAL "HIP")
        target_compile_definitions(simple_trainer PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
    elseif(GPU_RUNTIME STREQUAL "CUDA")
        target_compile_definitions(simple_trainer PRIVATE USE_CUDA)
    elseif(GPU_RUNTIME STREQUAL "MPS")
        target_compile_definitions(simple_trainer PRIVATE USE_MPS)
    endif()
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    file(GLOB OPENCV_DLL "${OPENCV_DIR}/x64/vc16/bin/opencv_world490.dll")
    set(DLLS_TO_COPY ${TORCH_DLLS} ${OPENCV_DLL})
    add_custom_command(TARGET opensplat
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLLS_TO_COPY}
        $<TARGET_FILE_DIR:opensplat>)
endif (MSVC)

add_compile_definitions(GLOG_USE_GLOG_EXPORT)
