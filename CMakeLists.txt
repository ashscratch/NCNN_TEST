cmake_minimum_required(VERSION 3.10)
project(NCNN_Deploy)

set(CMAKE_CXX_STANDARD 14)
#引入OPENCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#引入NCNN
set(NCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ncnn/build/install")
include_directories(${NCNN_DIR}/include/ncnn)
link_directories(${NCNN_DIR}/lib)

add_executable(deploy_test src/main.cpp)
#引入OPENMP并连接
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(deploy_test PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(deploy_test  PUBLIC ncnn ${OpenCV_LIBS} )