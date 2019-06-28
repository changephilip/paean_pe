cmake_minimum_required(VERSION 3.5)
project(CHIP)

set(CMAKE_CXX_STANDARD 11)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/home/qianjiaqiang/cub/cub -fstack-protector -D_FORCE_INLINES")

set(GPU_ARCHITECTURE "sm_61" CACHE STRING "Target GPU architecture")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-9.2")
set(CUDA_LIBRARY_DIR /usr/local/cuda-9.2/lib64/)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_DEBUG_FLAGS} -arch=${GPU_ARCHITECTURE} -use_fast_math -I/home/qianjiaqiang/cub/cub)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(CUDA 9.2 REQUIRED)
find_package(Thrust)
find_package(HTSlib)
find_package(Threads)
find_package(ZLIB REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)

set(CHIP_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(CUB_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/cub)
set(GFF_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/gff)
file(GLOB GFF_SRCS ${GFF_SOURCE_DIR}/src/*.cpp)

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${THRUST_INCLUDE_DIR}
        ${HTSlib_INCLUDE_DIR}
        ${CHIP_SOURCE_DIR}/include
        ${ZLIB_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${GFF_SOURCE_DIR}/include
        ${CUB_SOURCE_DIR}
)

cuda_add_executable(paean ${CHIP_SOURCE_DIR}/src/gene.cu ${GFF_SRCS} )
target_link_libraries(paean ${HTSlib_LIBRARY} ${ZLIB_LIBRARIES} ${Boost_LIBRARIES} Threads::Threads)
