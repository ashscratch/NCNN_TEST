cmake_minimum_required(VERSION 3.10)
project(NCNN_Deploy)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)

#寻找依赖
find_package(OpenCV REQUIRED)
find_package(OpenMP)
set(NCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ncnn/build/install")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)  # 寻找 inference_engine.h 和 thread_pool.h
include_directories(${NCNN_DIR}/include/ncnn)             # 寻找 net.h 等 NCNN 头文件
include_directories(${OpenCV_INCLUDE_DIRS})               # 寻找 opencv2/opencv.hpp

link_directories(${NCNN_DIR}/lib)

add_executable(deploy_single src/main_single.cpp src/inference_engine.cpp)
target_link_libraries(deploy_single PUBLIC ncnn ${OpenCV_LIBS})

add_executable(deploy_multi src/main_multi.cpp src/inference_engine.cpp)
target_link_libraries(deploy_multi PUBLIC ncnn ${OpenCV_LIBS})


if(OpenMP_CXX_FOUND)
    target_link_libraries(deploy_single PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(deploy_multi PUBLIC OpenMP::OpenMP_CXX)
endif()
