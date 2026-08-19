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

