# Copyright (C) 2019-2020 IBM Corp.
# This program is Licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. See accompanying LICENSE file.
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

## Use -std=c++17 as default.
set(CMAKE_CXX_STANDARD 17)
## Disable C++ extensions
set(CMAKE_CXX_EXTENSIONS OFF)
## Require full C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_PREFIX_PATH ~/helib_install/helib_pack/share/cmake/helib)
set(CMAKE_PREFIX_PATH /usr/local/share/cmake/helib)

project(BGV_packed_arithmetic
       	LANGUAGES CUDA CXX)

set(CMAKE_PREFIX_PATH ~/helib_install/helib_pack/share/cmake/helib)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include_directories(${PROJECT_SOURCE_DIR})
        
# target_include_directories(compare_cuda PUBLIC /usr/local/include)




set(CUDA_SEPARABLE_COMPILATION ON)
include_directories(/usr/local/include)
# include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("/usr/local/cuda/include")
find_package(CUDA  REQUIRED)
# find_package(CUDA_ACCEL REQUIRED)
find_package(helib REQUIRED)
set (CMAKE_CXX_FLAGS "-g")
set(CUDA_NVCC_FLAGS "-G -g")
set(CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS})

add_executable(BGV_packed_arithmetic BGV_packed_arithmetic.cpp)

target_link_libraries(BGV_packed_arithmetic helib CUDA_ACCEL)
# target_link_libraries(BGV_packed_arithmetic helib)
