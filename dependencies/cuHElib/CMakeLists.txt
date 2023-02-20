project(CUDA_ACCEL
        LANGUAGES CUDA)
include(ExternalProject)
cmake_minimum_required(VERSION 3.22)
# include(ExternalProject)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_STANDARD 17)


# # Path of the gmp library file that will be installed (libgmp.so/dylib)
# set(cuHElib
#     "${}"
# )

# target_include_directories(${PROJECT_NAME} PUBLIC "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/dependencies/cuHElib")
add_library(CUDA_ACCEL STATIC
gpu_accel.cu
gpu_accel.cuh
)

# find_package(CUDA QUIET)
# if(CUDA_FOUND)
#     include_directories(${CUDA_INCLUDE_DIRS})
#     SET(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
#     SET(LIBS ${LIBS} ${ALL_CUDA_LIBS})
#     message(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS} ${ALL_CUDA_LIBS}")
#     set(CUDA_PROPAGATE_HOST_FLAGS ON)
#     set(CUDA_SEPARABLE_COMPILATION ON)
#     list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_30,code=compute_30 -rdc=true -c )
#     list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52 -rdc=true -c )
# endif()

# target_include_directories(compare_cuda PUBLIC ~/helib_install/helib_pack/include)

# include_directories(~/helib_install/helib_pack/lib)



find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set_target_properties(CUDA_ACCEL PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON
                            CUDA_RESOLVE_DEVICE_SYMBOLS OFF)


# find_package(helib REQUIRED)
set(CMAKE_CXX_FLAGS "-g")
set(CUDA_NVCC_FLAGS "-G -g -rdc=true -c -lcufft")
set(CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS})
set_target_properties(CUDA_ACCEL PROPERTIES CUDA_ARCHITECTURES "35;50;72")
