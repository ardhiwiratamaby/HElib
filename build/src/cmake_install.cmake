# Install script for directory: /home/ardhy/Documents/research/new_project/bgv-comparison/HElib/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/build/lib/libhelib.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/helib" TYPE FILE FILES
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/helib.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/apiAttributes.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/ArgMap.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/binaryArith.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/binaryCompare.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/bluestein.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/ClonedPtr.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/CModulus.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/CtPtrs.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/Ctxt.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/debugging.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/DoubleCRT.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/EncryptedArray.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/EvalMap.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/Context.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/FHE.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/keys.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/keySwitching.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/log.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/hypercube.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/IndexMap.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/IndexSet.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/intraSlot.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/JsonWrapper.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/matching.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/matmul.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/Matrix.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/multicore.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/norms.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/NumbTh.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/PAlgebra.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/partialMatch.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/permutations.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/polyEval.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/PolyMod.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/PolyModRing.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/powerful.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/primeChain.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/PtrMatrix.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/PtrVector.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/Ptxt.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/randomMatrices.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/range.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/recryption.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/replicate.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/sample.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/scheme.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/set.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/SumRegister.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/tableLookup.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/timing.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/zzX.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/assertions.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/exceptions.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/PGFFT.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/fhe_stats.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/zeroValue.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/include/helib/EncodedPtxt.h"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/build/src/helib/version.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/helib/helibTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/helib/helibTargets.cmake"
         "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/build/src/CMakeFiles/Export/share/cmake/helib/helibTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/helib/helibTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/helib/helibTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/helib" TYPE FILE FILES "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/build/src/CMakeFiles/Export/share/cmake/helib/helibTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/helib" TYPE FILE FILES "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/build/src/CMakeFiles/Export/share/cmake/helib/helibTargets-debug.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/helib" TYPE FILE FILES
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/build/src/helibConfig.cmake"
    "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/build/src/helibConfigVersion.cmake"
    )
endif()

