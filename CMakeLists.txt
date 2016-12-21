cmake_minimum_required(VERSION 2.6)
project(mf)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -Wall")
#set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -flto -g -Wall")
#set(CMAKE_CXX_FLAGS "-std=c++11  -g -Wall")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")          
set(CMAKE_FILES_DIRECTORY "build/")    

include_directories(/Users/mohitsharma/dev/gklib/trunk)
include_directories(/Users/mohitsharma/dev/eigen)
include_directories(/Users/mohitsharma/dev/SVDLIBC)

link_directories(/Users/mohitsharma/dev/gklib/trunk/build/Darwin-x86_64)
link_directories(/Users/mohitsharma/dev/SVDLIBC)

file(GLOB SOURCES "*.cpp")
add_executable(mf ${SOURCES})

set(gflags_DIR "/Users/mohitsharma/dev/gflags/build")
find_package(gflags REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

target_link_libraries(mf GKlib svd m gflags)

