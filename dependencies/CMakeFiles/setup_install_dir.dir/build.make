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
CMAKE_SOURCE_DIR = /home/ardhy/Documents/research/new_project/bgv-comparison/HElib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ardhy/Documents/research/new_project/bgv-comparison/HElib

# Utility rule file for setup_install_dir.

# Include any custom commands dependencies for this target.
include dependencies/CMakeFiles/setup_install_dir.dir/compiler_depend.make

# Include the progress variables for this target.
include dependencies/CMakeFiles/setup_install_dir.dir/progress.make

setup_install_dir: dependencies/CMakeFiles/setup_install_dir.dir/build.make
	cd /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/dependencies && /home/ardhy/anaconda2/lib/python2.7/site-packages/cmake/data/bin/cmake -E make_directory /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/helib_pack/lib
.PHONY : setup_install_dir

# Rule to build all files generated by this target.
dependencies/CMakeFiles/setup_install_dir.dir/build: setup_install_dir
.PHONY : dependencies/CMakeFiles/setup_install_dir.dir/build

dependencies/CMakeFiles/setup_install_dir.dir/clean:
	cd /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/dependencies && $(CMAKE_COMMAND) -P CMakeFiles/setup_install_dir.dir/cmake_clean.cmake
.PHONY : dependencies/CMakeFiles/setup_install_dir.dir/clean

dependencies/CMakeFiles/setup_install_dir.dir/depend:
	cd /home/ardhy/Documents/research/new_project/bgv-comparison/HElib && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ardhy/Documents/research/new_project/bgv-comparison/HElib /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/dependencies /home/ardhy/Documents/research/new_project/bgv-comparison/HElib /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/dependencies /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/dependencies/CMakeFiles/setup_install_dir.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dependencies/CMakeFiles/setup_install_dir.dir/depend

