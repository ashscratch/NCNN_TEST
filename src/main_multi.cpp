#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>
#include "inference_engine.h"
#include "thread_pool.h"

class FrameResult{
public:
    cv::Mat frame;
    int class_index;
    float confidence;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
};

int main(){
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
    
    Thread_pool pool(4);
    std::queue<std::future<FrameResult>> pending_tasks;
    int max_pipeline_depth = 4; //流水线最大深处

    cv::Mat frame;
    std::cout << "多线程推理流水线" << std::endl;

    int total_frames_processed = 0;
    auto pipeline_start_time = std::chrono::high_resolution_clock::now();

    cv::namedWindow("Multi-Thread Inference", cv::WINDOW_NORMAL);
    cv::resizeWindow("Multi-Thread Inference", 1200, 800);

    auto last_display_time = std::chrono::high_resolution_clock::now();

    while(true){
        cap >> frame;
        if(frame.empty()) break;
        cv::Mat thread_safe_frame = frame.clone();
        auto start_time = std::chrono::high_resolution_clock::now();

        auto future_result = pool.enqueue([&engine, thread_safe_frame, start_time]() -> FrameResult{
            float conf = 0.0f;
            int id = engine.Infer(thread_safe_frame, conf);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return {thread_safe_frame, id, conf, start_time};
        });

        pending_tasks.push(std::move(future_result));

        if(pending_tasks.size() >= max_pipeline_depth) {
            FrameResult result = pending_tasks.front().get();
            pending_tasks.pop();
            total_frames_processed++;

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - last_display_time);
            last_display_time = end_time;
            float fps = 1000000.f / (duration.count() + 1e-5);

            if(result.confidence > 0.6f){
            std::string label = "ID" + std::to_string(result.class_index) + "    " +"Conf" + std::to_string(result.confidence) .substr(0,4); //置信度取4位
            cv::putText(result.frame, label, cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 255, 0), 3); //放置标签
            }

            std::string fps_label = "FPS: " + std::to_string(fps).substr(0, 4);
            cv::putText(result.frame, fps_label, cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(255, 0, 0), 3);

            cv::imshow("Multi-Thread Inference", result.frame);

            if ((cv::waitKey(1) & 0xFF) == 27) {
                break;
            }
        }
    }

    while (!pending_tasks.empty()) {
        FrameResult result = pending_tasks.front().get();
        pending_tasks.pop();
        total_frames_processed++;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - last_display_time);
        last_display_time = end_time;
        float fps = 1000000.f / (duration.count() + 1e-5);
    
        if(result.confidence > 0.6f){
            std::string label = "ID" + std::to_string(result.class_index) + "    " +"Conf" + std::to_string(result.confidence) .substr(0,4); //置信度取4位
            cv::putText(result.frame, label, cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 255, 0), 3); //放置标签
        }

        std::string fps_label = "FPS: " + std::to_string(fps).substr(0, 4);
        cv::putText(result.frame, fps_label, cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(255, 0, 0), 3);

        cv::imshow("Multi-Thread Inference", result.frame);
        cv::waitKey(1); 
    }

    cap.release();
    cv::destroyAllWindows(); 

    auto pipeline_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pipeline_end_time - pipeline_start_time);
    float total_seconds = total_duration.count() / 1000.0f;
    float average_fps = total_frames_processed / total_seconds;
    std::cout << average_fps << std::endl;
    return 0;
}