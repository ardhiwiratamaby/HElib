# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/ardhy/anaconda2/lib/python2.7/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/ardhy/anaconda2/lib/python2.7/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic

# Include any dependencies generated for this target.
include CMakeFiles/BGV_packed_arithmetic.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/BGV_packed_arithmetic.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/BGV_packed_arithmetic.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BGV_packed_arithmetic.dir/flags.make

CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.o: CMakeFiles/BGV_packed_arithmetic.dir/flags.make
CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.o: BGV_packed_arithmetic.cpp
CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.o: CMakeFiles/BGV_packed_arithmetic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.o -MF CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.o.d -o CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.o -c /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic/BGV_packed_arithmetic.cpp

CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic/BGV_packed_arithmetic.cpp > CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.i

CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic/BGV_packed_arithmetic.cpp -o CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.s

# Object files for target BGV_packed_arithmetic
BGV_packed_arithmetic_OBJECTS = \
"CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.o"

# External object files for target BGV_packed_arithmetic
BGV_packed_arithmetic_EXTERNAL_OBJECTS =

BGV_packed_arithmetic: CMakeFiles/BGV_packed_arithmetic.dir/BGV_packed_arithmetic.cpp.o
BGV_packed_arithmetic: CMakeFiles/BGV_packed_arithmetic.dir/build.make
BGV_packed_arithmetic: /usr/local/lib/libhelib.a
BGV_packed_arithmetic: /usr/local/lib/libntl.so
BGV_packed_arithmetic: /usr/local/lib/libgmp.so
BGV_packed_arithmetic: CMakeFiles/BGV_packed_arithmetic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable BGV_packed_arithmetic"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BGV_packed_arithmetic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BGV_packed_arithmetic.dir/build: BGV_packed_arithmetic
.PHONY : CMakeFiles/BGV_packed_arithmetic.dir/build

CMakeFiles/BGV_packed_arithmetic.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BGV_packed_arithmetic.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BGV_packed_arithmetic.dir/clean

CMakeFiles/BGV_packed_arithmetic.dir/depend:
	cd /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/BGV_packed_arithmetic/CMakeFiles/BGV_packed_arithmetic.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/BGV_packed_arithmetic.dir/depend

