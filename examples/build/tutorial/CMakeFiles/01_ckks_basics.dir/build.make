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
CMAKE_SOURCE_DIR = /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build

# Include any dependencies generated for this target.
include tutorial/CMakeFiles/01_ckks_basics.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tutorial/CMakeFiles/01_ckks_basics.dir/compiler_depend.make

# Include the progress variables for this target.
include tutorial/CMakeFiles/01_ckks_basics.dir/progress.make

# Include the compile flags for this target's objects.
include tutorial/CMakeFiles/01_ckks_basics.dir/flags.make

tutorial/CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.o: tutorial/CMakeFiles/01_ckks_basics.dir/flags.make
tutorial/CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.o: ../tutorial/01_ckks_basics.cpp
tutorial/CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.o: tutorial/CMakeFiles/01_ckks_basics.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tutorial/CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.o"
	cd /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build/tutorial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tutorial/CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.o -MF CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.o.d -o CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.o -c /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/tutorial/01_ckks_basics.cpp

tutorial/CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.i"
	cd /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build/tutorial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/tutorial/01_ckks_basics.cpp > CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.i

tutorial/CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.s"
	cd /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build/tutorial && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/tutorial/01_ckks_basics.cpp -o CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.s

# Object files for target 01_ckks_basics
01_ckks_basics_OBJECTS = \
"CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.o"

# External object files for target 01_ckks_basics
01_ckks_basics_EXTERNAL_OBJECTS =

bin/01_ckks_basics: tutorial/CMakeFiles/01_ckks_basics.dir/01_ckks_basics.cpp.o
bin/01_ckks_basics: tutorial/CMakeFiles/01_ckks_basics.dir/build.make
bin/01_ckks_basics: /usr/local/lib/libhelib.a
bin/01_ckks_basics: /usr/local/lib/libntl.so
bin/01_ckks_basics: /usr/local/lib/libgmp.so
bin/01_ckks_basics: tutorial/CMakeFiles/01_ckks_basics.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/01_ckks_basics"
	cd /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build/tutorial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/01_ckks_basics.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tutorial/CMakeFiles/01_ckks_basics.dir/build: bin/01_ckks_basics
.PHONY : tutorial/CMakeFiles/01_ckks_basics.dir/build

tutorial/CMakeFiles/01_ckks_basics.dir/clean:
	cd /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build/tutorial && $(CMAKE_COMMAND) -P CMakeFiles/01_ckks_basics.dir/cmake_clean.cmake
.PHONY : tutorial/CMakeFiles/01_ckks_basics.dir/clean

tutorial/CMakeFiles/01_ckks_basics.dir/depend:
	cd /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/tutorial /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build/tutorial /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/examples/build/tutorial/CMakeFiles/01_ckks_basics.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tutorial/CMakeFiles/01_ckks_basics.dir/depend

