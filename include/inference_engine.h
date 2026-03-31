#pragma once
#include <opencv2/opencv.hpp>
#include "net.h"
#include <string>

class InferenceEngine {
private:
    //核心骨架：生命周期内只初始化一次，只读全局共享
    ncnn::Net net_; 

    //模型要求
    const int target_size = 227;
    const float mean_vals[3] = {104.f, 117.f, 123.f};

public:
    InferenceEngine();
    ~InferenceEngine();

    //加载模型
    bool LoadModel(const std::string& param_path, const std::string& bin_path);

    //执行推理：
    int Infer(const cv::Mat& frame, float& out_confidence);
};