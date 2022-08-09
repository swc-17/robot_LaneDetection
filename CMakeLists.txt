cmake_minimum_required(VERSION 2.6)

project(lane_det)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable(model_conversion ${PROJECT_SOURCE_DIR}/src/model_conversion.cpp)
target_link_libraries(model_conversion nvinfer ${CUDA_LIBRARIES} ${OpenCV_LIBS})

add_executable(main ${PROJECT_SOURCE_DIR}/src/UFLD.cpp ${PROJECT_SOURCE_DIR}/src/main.cpp )
target_link_libraries(main nvinfer ${CUDA_LIBRARIES} ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

