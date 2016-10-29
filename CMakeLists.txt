cmake_minimum_required(VERSION 2.6)
project(mf)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -g -Wall")
#set(CMAKE_CXX_FLAGS "-std=c++11  -g -Wall")
include_directories(/home/grad02/mohit/George/GKlib/trunk)
include_directories(/home/grad02/mohit/lib/eigen)
include_directories(/home/grad02/mohit/lib/SVDLIBC)

link_directories(/home/grad02/mohit/George/GKlib/trunk/build/Linux-x86_64/)
link_directories(/home/grad02/mohit/lib/SVDLIBC)

file(GLOB SOURCES "*.cpp")
add_executable(mf ${SOURCES})

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

target_link_libraries(mf GKlib svd m)

