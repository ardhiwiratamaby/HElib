cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(MatMul)

## Use -std=c++17 as default.
set(CMAKE_CXX_STANDARD 17)
## Disable C++ extensions
set(CMAKE_CXX_EXTENSIONS OFF)
## Require full C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH /usr/local/share/cmake/helib)
include_directories(/usr/local/include)
include_directories(/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/cmake)
set(CMAKE_PREFIX_PATH /usr/local/share/cmake/helib)
list(APPEND CMAKE_MODULE_PATH  /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/cmake)
find_package(helib REQUIRED)
FIND_PACKAGE(NTL REQUIRED)
#FIND_PACKAGE(GMP REQUIRED)

add_executable(MatMul blockMatmul.cpp blockMatmul1D.cpp matmul.cpp matmul1D.cpp Test_matmul.cpp) 
target_link_libraries(MatMul PUBLIC helib)
