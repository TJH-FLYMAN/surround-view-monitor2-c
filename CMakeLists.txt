cmake_minimum_required (VERSION 3.0)

project(avm_main)
add_executable (avm_main main.cpp)
add_executable (calib_main main.cpp ./srcs/common.cpp)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "opencv inc path " ${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/srcs)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

target_link_libraries(avm_main ${OpenCV_LIBS})
target_link_libraries(calib_main ${OpenCV_LIBS})
