#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"

int main()
{
    ncnn::Net squeezenet;
    squeezenet.opt.num_threads = 4;

    //导入网络模型&检查是否成功
    int param_status = squeezenet.load_param("../assets/squeezenet_v1.1.param");
    int bin_status = squeezenet.load_model("../assets/squeezenet_v1.1.bin");
    if(param_status != 0){
        std::cerr << "错误：无法导入拓扑结构" << std::endl;
        return -1;
    }
    if(bin_status != 0){
        std::cerr << "错误：无法导入权重数据" << std::endl;
        return -1;
    }
    std::cout << "模型加载成功" << std::endl;

    //图像读取和预处理
    cv::Mat img = cv::imread("../assets/test.jpg", cv::IMREAD_COLOR);
    if(img.empty()){
        std::cerr << "加载图片失败" << std::endl;
        return -1;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, 227,227);//img.data uchar*指针  uchar->float
    const float mean_val[3] = {104.f, 117.f, 123.f}; //储存减去的·BGR均值
    in.substract_mean_normalize(mean_val, 0); //减去均值，0代表不除以方差

    ncnn::Extractor ex = squeezenet.create_extractor(); //创建前向传播处理器
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

    std::cout << "推理完成" << std::endl;
    std::cout << "预测类别索引： " << max_index << std::endl;
    std::cout << "置信度： " << max_prob << std::endl;



    return 0;
}