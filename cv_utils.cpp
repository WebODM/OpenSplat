#include "cv_utils.hpp"

cv::Mat imreadRGB(const std::string &filename){
    cv::Mat cImg = cv::imread(filename);

    if (cImg.empty()){
        std::cerr << "Cannot read " << filename << std::endl
                  << "Make sure the path to your images is correct" << std::endl;
        exit(1);
    }

    cv::cvtColor(cImg, cImg, cv::COLOR_BGR2RGB);
    return cImg;
}

cv::Mat imreadMask(const std::string &filename){
    cv::Mat mask = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (mask.empty()){
        std::cerr << "Cannot read mask " << filename << std::endl;
        exit(1);
    }
    return mask;
}

void imwriteRGB(const std::string &filename, const cv::Mat &image){
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_RGB2BGR);
    cv::imwrite(filename, rgb);
}

cv::Mat floatNxNtensorToMat(const torch::Tensor &t){
    return cv::Mat(t.size(0), t.size(1), CV_32F, t.data_ptr());
}

torch::Tensor floatNxNMatToTensor(const cv::Mat &m){
    return torch::from_blob(m.data, { m.rows, m.cols }, torch::kFloat32).clone();
}

cv::Mat tensorToImage(const torch::Tensor &t){
    int h = t.sizes()[0];
    int w = t.sizes()[1];
    int c = t.sizes()[2];

    int type = CV_8UC3;
    if (c != 3) throw std::runtime_error("Only images with 3 channels are supported");

    cv::Mat image(h, w, type);
    torch::Tensor scaledTensor = (t * 255.0).toType(torch::kU8);
    uint8_t* dataPtr = static_cast<uint8_t*>(scaledTensor.data_ptr());
    std::copy(dataPtr, dataPtr + (w * h * c), image.data);

    return image;
}

torch::Tensor imageToTensor(const cv::Mat &image){
    torch::Tensor img = torch::from_blob(image.data, { image.rows, image.cols, image.dims + 1 }, torch::kU8);
    return (img.toType(torch::kFloat32) / 255.0f);
}

torch::Tensor maskToTensor(const cv::Mat &mask){
    torch::Tensor m = torch::from_blob(mask.data, { mask.rows, mask.cols, 1 }, torch::kU8);
    
    // Binary mask: threshold at 128, output 0.0 or 1.0
    return (m.toType(torch::kFloat32) / 255.0f).ge(0.5f).toType(torch::kFloat32);
}

cv::Mat tensorToMask(const torch::Tensor &t){
    int h = t.sizes()[0];
    int w = t.sizes()[1];

    cv::Mat mask(h, w, CV_8UC1);
    torch::Tensor scaledTensor = (t.squeeze() * 255.0).toType(torch::kU8);
    uint8_t* dataPtr = static_cast<uint8_t*>(scaledTensor.data_ptr());
    std::copy(dataPtr, dataPtr + (w * h), mask.data);

    return mask;
}

