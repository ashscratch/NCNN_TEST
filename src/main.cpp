#include <iostream>
#include <opencv2/opencv.hpp>
#include "inference_engine.h"

int main()
{
    InferenceEngine engine;
    if(!engine.LoadModel("../assets/squeezenet_v1.1.param", "../assets/squeezenet_v1.1.bin")){
        return -1;
    }
    std::cout << "模型初始化成功" << std::endl;

    cv::Mat test_frame = cv::imread("../assets/test.jpg", cv::IMREAD_COLOR);
    float confidence = 0.f;
    int index = engine.Infer(test_frame, confidence);


    std::cout << "推理完成" << std::endl;
    std::cout << "预测类别索引： " << index << std::endl;
    std::cout << "置信度： " << confidence << std::endl;

    return 0;
}