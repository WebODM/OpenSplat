#ifndef CV_UTILS
#define CV_UTILS

#include <torch/torch.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat imreadRGB(const std::string &filename);
cv::Mat tensorToImage(const torch::Tensor &t);
torch::Tensor imageToTensor(const cv::Mat &image);

#endif