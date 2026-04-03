#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include "inference_engine.h"

int main()
{
    InferenceEngine engine;
    if(!engine.LoadModel("../assets/squeezenet_v1.1.param", "../assets/squeezenet_v1.1.bin")){
        return -1;
    }
    std::cout << "模型初始化成功" << std::endl;

    cv::VideoCapture cap("../assets/demo.mp4");
    if(!cap.isOpened()){
        std::cerr << "错误：视频流打开失败" << std::endl;
        return -1;
    }

    cv::Mat frame;
    float confidence =0.0f;
    std::cout << "开始进行实时推理... " << std::endl << "按下ESC键可退出" << std::endl;

    int total_frames_processed = 0;
    auto pipeline_start_time = std::chrono::high_resolution_clock::now();

    // 创建一个可调整大小的窗口
    cv::namedWindow("NCNN Edge Deployment Demo", cv::WINDOW_NORMAL);
    cv::resizeWindow("NCNN Edge Deployment Demo", 1200, 800);

    //循环处理视频流
    while(true){
        //开始时间
        auto start_time = std::chrono::high_resolution_clock::now();
        
        //抓取帧
        cap >> frame;
        if(frame.empty()){
            std::cout << "视频流故障或已结束" << std::endl;
            break;
        }

        //执行推理
        int class_id = engine.Infer(frame, confidence);
        total_frames_processed++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        //记录结束时间和耗时
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        //计算FPS
        float fps = 1000000.f / (duration.count() + 1e-5);

        //结果可视化（置信度大于0.6时）
        if(confidence > 0.6f){
            std::string label = "Class" + std::to_string(class_id) + "Conf" + std::to_string(confidence) .substr(0,4); //置信度取4位
            cv::putText(frame, label, cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 255, 0), 3); //放置标签
        }

        //标注FPS
        std::string fps_label = "FPS: " + std::to_string(fps).substr(0, 4);
        cv::putText(frame, fps_label, cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(255, 0, 0), 3);

        cv::imshow("NCNN Edge Deployment Demo", frame);

        // 等待并判断是否退出
        if((cv::waitKey(1) & 0xFF) == 27){
            break;
        }
    }
    std::cout << "退出推理" << std::endl;

    //释放硬件&销毁窗口
    cap.release();
    cv::destroyAllWindows();

    auto pipeline_end_time = std::chrono::high_resolution_clock::now();
    
    // 计算总耗时 (毫秒)
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end_time - pipeline_start_time);
    float total_seconds = total_duration.count() / 1000.0f;
    
    // 计算平均 FPS (总帧数 / 总秒数)
    float average_fps = total_frames_processed / total_seconds;
    std::cout << average_fps << std::endl;

    return 0;
}