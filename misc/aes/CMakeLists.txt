cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(AES)
set(CMAKE_PREFIX_PATH /usr/local/share/cmake/helib)
include_directories(/usr/local/include)
find_package(helib REQUIRED)

add_executable(TEST_AES simpleAES.cpp homAES.cpp Test_AES.cpp)
target_link_libraries(TEST_AES PUBLIC helib)
