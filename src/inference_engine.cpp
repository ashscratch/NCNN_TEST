#include "inference_engine.h"
#include <iostream>

//没有返回值，为了防止出现异常
InferenceEngine::InferenceEngine(){  
}

InferenceEngine::~InferenceEngine(){  
    net_.clear();
}

bool InferenceEngine::LoadModel(const std::string& param_path, const std::string& bin_path){
    if(net_.load_param(param_path.c_str()) != 0){
        std::cerr << "加载param失败" << param_path << std::endl;
        return false;
    }
     if(net_.load_model(bin_path.c_str()) != 0){
        std::cerr << "加载bin失败" << bin_path << std::endl;
        return false;
    }
    return true;
}

int InferenceEngine::Infer(const cv::Mat& frame, float& out_confidence){
    if(frame.empty()){
        std::cerr << "传入视频流为空" << std::endl;
        return -1;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data, ncnn::Mat::PIXEL_BGR, 
        frame.cols, frame.rows, target_size, target_size);//frame.data uchar*指针  uchar->float
    in.substract_mean_normalize(mean_vals, 0);

    //每次调用函数都会生成提取器
    ncnn::Extractor ex = net_.create_extractor(); //创建前向传播处理器
    net_.opt.num_threads = 1;
    ex.input("data", in); //将张量in绑定到输入节点data->来自param文件
    ncnn::Mat out;
    ex.extract("prob", out);

    float* out_data = (float*) out.data; //提取结果并转化为浮点数
    int max_index = 0;
    float max_prob = 0.0f;
    for(int i = 0; i < out.w; i++){
        if(out_data[i] > max_prob){
            max_prob = out_data[i];
            max_index = i;
        }
    }

    out_confidence = max_prob;
    return max_index;
}