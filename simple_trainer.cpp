#include <iostream>
#include <cmath>
#include <filesystem>

#include <torch/torch.h>
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#elif defined(USE_CUDA)
#include <torch/cuda.h>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "project_gaussians.hpp"
#include "rasterize_gaussians.hpp"
#include "constants.hpp"
#include "cv_utils.hpp"
#include <cxxopts.hpp>

using namespace torch::indexing;
namespace fs = std::filesystem;

int main(int argc, char **argv){
    cxxopts::Options options("simple_trainer", "Test program for gsplat execution - " APP_VERSION);
    options.add_options()
        ("cpu", "Force CPU execution")
        ("width", "Test image width", cxxopts::value<int>()->default_value("256"))
        ("height", "Test image height", cxxopts::value<int>()->default_value("256"))
        ("iters", "Number of iterations", cxxopts::value<int>()->default_value("1000"))
        ("points", "Number of gaussians", cxxopts::value<int>()->default_value("100000"))
        ("lr", "Learning rate", cxxopts::value<float>()->default_value("0.01"))
        ("render", "Save rendered images to folder", cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Print usage")
        ("version", "Print version")
        ;
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    if (result.count("version")) {
        std::cout << APP_VERSION << std::endl;
        return EXIT_SUCCESS;
    }

    int width = result["width"].as<int>(),
        height = result["height"].as<int>();
    int numPoints = result["points"].as<int>();
    int iterations = result["iters"].as<int>();
    float learningRate = result["lr"].as<float>();
    std::string render = result["render"].as<std::string>();
    if (!render.empty() && !fs::exists(render)) fs::create_directories(render);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available() && result.count("cpu") == 0){
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
    }else if(torch::mps::is_available() && result.count("cpu") == 0){
        std::cout << "Using MPS" << std::endl;
        device = torch::kMPS;
    }else{
        std::cout << "Using CPU" << std::endl;
    }

    // Test image
    // Top left red
    // Bottom right blue
    torch::Tensor gtImage = torch::ones({height, width, 3});
    gtImage.index_put_({Slice(None, height / 2), Slice(None, width / 2), Slice()}, torch::tensor({1.0, 0.0, 0.0}));
    gtImage.index_put_({Slice(height / 2, None), Slice(width / 2, None), Slice()}, torch::tensor({0.0, 0.0, 1.0}));

    // cv::Mat image = tensorToImage(gtImage);
    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    // cv::imwrite("test.png", image);

    gtImage = gtImage.to(device);
    double fovX = PI / 2.0; // horizontal field of view (90 deg)
    double focal = 0.5 * static_cast<double>(width) / std::tan(0.5 * fovX);

    TileBounds tileBounds = std::make_tuple((width + BLOCK_X - 1) / BLOCK_X,
                      (height + BLOCK_Y - 1) / BLOCK_Y,
                      1);
    
    // Init gaussians
#ifdef USE_CUDA
    torch::cuda::manual_seed_all(0);
#endif
    torch::manual_seed(0);

    // Random points, scales and colors
    torch::Tensor means = 2.0 * (torch::rand({numPoints, 3}, torch::kCPU) - 0.5); // Positions [-1, 1]
    torch::Tensor scales = torch::rand({numPoints, 3}, torch::kCPU);
    // torch::Tensor means = torch::tensor({{0.5f, 0.5f, -5.0f}, {0.5f, 0.5f, -6.0f}, {0.25f, 0.25f, -4.0f}}, torch::kCPU);
    // torch::Tensor scales = torch::tensor({{0.5f, 0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}, torch::kCPU);
    torch::Tensor rgbs = torch::rand({numPoints, 3}, torch::kCPU);
    
    // Random rotations (quaternions)
    // quats = ( sqrt(1-u) sin(2πv), sqrt(1-u) cos(2πv), sqrt(u) sin(2πw), sqrt(u) cos(2πw))
    torch::Tensor u = torch::rand({numPoints, 1}, torch::kCPU);
    torch::Tensor v = torch::rand({numPoints, 1}, torch::kCPU);
    torch::Tensor w = torch::rand({numPoints, 1}, torch::kCPU);

    means = means.to(device);
    scales = scales.to(device);
    rgbs = rgbs.to(device);
    u = u.to(device);
    v = v.to(device);
    w = w.to(device);    

    torch::Tensor quats = torch::cat({
                torch::sqrt(1.0 - u) * torch::sin(2.0 * PI * v),
                torch::sqrt(1.0 - u) * torch::cos(2.0 * PI * v),
                torch::sqrt(u) * torch::sin(2.0 * PI * w),
                torch::sqrt(u) * torch::cos(2.0 * PI * w),
            }, -1);
    
    torch::Tensor opacities = torch::ones({numPoints, 1}, device);

    // View matrix (translation in Z by 8 units)
    torch::Tensor viewMat = torch::tensor({
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 8.0},
            {0.0, 0.0, 0.0, 1.0}
        }, device);

    torch::Tensor background = torch::zeros(gtImage.size(2), device);

    means.requires_grad_();
    scales.requires_grad_();
    quats.requires_grad_();
    rgbs.requires_grad_();
    opacities.requires_grad_();

    torch::optim::Adam optimizer({rgbs, means, scales, opacities, quats}, learningRate);
    torch::nn::MSELoss mseLoss;
    torch::Tensor outImg;

    for (size_t i = 0; i < iterations; i++){
        if (device == torch::kCPU){
            auto p = ProjectGaussiansCPU::apply(means, scales, 1, 
                                quats, viewMat, viewMat,
                                focal, focal,
                                width / 2,
                                height / 2,
                                height,
                                width);
            
            outImg = RasterizeGaussiansCPU::apply(
                p[0], // xys
                p[1], // radii,
                p[2], // conics
                torch::sigmoid(rgbs),
                torch::sigmoid(opacities),
                p[3], // cov2d
                p[4], // camDepths
                height,
                width,
                background)[0];
        }else{
            #if defined(USE_HIP) || defined(USE_CUDA) || defined(USE_MPS)
                auto p = ProjectGaussians::apply(means, scales, 1, 
                                        quats, viewMat, viewMat,
                                        focal, focal,
                                        width / 2,
                                        height / 2,
                                        height,
                                        width,
                                        tileBounds);

                outImg = RasterizeGaussians::apply(
                    p[0], // xys
                    p[1], // depths
                    p[2], // radii,
                    p[3], // conics
                    p[4], // numTilesHit
                    torch::sigmoid(rgbs),
                    torch::sigmoid(opacities),
                    height,
                    width,
                    background)[0];
            #else
                throw std::runtime_error("GPU support not built, use --cpu");
            #endif
        }

        outImg.requires_grad_();
        torch::Tensor loss = mseLoss(outImg, gtImage);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::cout << "Iteration " << std::to_string(i + 1) << "/" << std::to_string(iterations) << " Loss: " << loss.item<float>() << std::endl; 
        
        if (!render.empty()){
            cv::Mat image = tensorToImage(outImg.detach().cpu());
            cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
            cv::imwrite((fs::path(render) / (std::to_string(i + 1) + ".png")).string(), image);
        }
    }
}