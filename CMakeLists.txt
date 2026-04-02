cmake_minimum_required(VERSION 3.10)
project(NCNN_Deploy)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)

#寻找依赖
find_package(OpenCV REQUIRED)
find_package(OpenMP)
set(NCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ncnn/build/install")

add_executable(deploy_test src/main.cpp src/inference_engine.cpp)

target_include_directories(deploy_test PRIVATE
    ${OpenCV_INCLUDE_DIRS}                # OpenCV 头文件
    ${NCNN_DIR}/include/ncnn              # ncnn 头文件
    ${CMAKE_CURRENT_SOURCE_DIR}/include   # 项目自有头文件
)
target_link_directories(deploy_test PRIVATE ${NCNN_DIR}/lib)
target_link_libraries(deploy_test PRIVATE
    ncnn                                 
    ${OpenCV_LIBS}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(deploy_test PRIVATE OpenMP::OpenMP_CXX)
endif()
