#ifndef INPUTDATA_H
#define INPUTDATA_H

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <opencv2/calib3d.hpp>
#include <torch/torch.h>

enum CameraType { Perspective };
struct Camera{
    int id = -1;
    int width = 0;
    int height = 0;
    float fx = 0;
    float fy = 0;
    float cx = 0;
    float cy = 0;
    float k1 = 0;
    float k2 = 0;
    float k3 = 0;
    float k4 = 0;
    float k5 = 0;
    float k6 = 0;
    float p1 = 0;
    float p2 = 0;
    torch::Tensor camToWorld;
    std::string filePath = "";
    std::string maskPath = "";
    bool invertMask = false;
    CameraType cameraType = CameraType::Perspective;

    Camera(){};
    Camera(int width, int height, float fx, float fy, float cx, float cy, 
        float k1, float k2, float k3, float p1, float p2,
        const torch::Tensor &camToWorld, const std::string &filePath) : 
        width(width), height(height), fx(fx), fy(fy), cx(cx), cy(cy), 
        k1(k1), k2(k2), k3(k3), p1(p1), p2(p2),
        camToWorld(camToWorld), filePath(filePath) {}
    torch::Tensor getIntrinsicsMatrix();
    bool hasDistortionParameters();
    torch::Tensor getImage(int downscaleFactor);
    torch::Tensor getMask(int downscaleFactor);
    torch::Tensor getEdgeMap(int downscaleFactor);
    bool hasMask() const { return mask.numel() > 0; }

    void loadImage(float downscaleFactor);
    torch::Tensor K;
    torch::Tensor image;
    torch::Tensor mask; // [H,W] float 0/1, aligned with image

    std::unordered_map<int, torch::Tensor> imagePyramids;
    std::unordered_map<int, torch::Tensor> maskPyramids;
    std::unordered_map<int, torch::Tensor> edgePyramids;
};

struct Points{
    torch::Tensor xyz;
    torch::Tensor rgb;
};
struct InputData{
    std::vector<Camera> cameras;
    float scale;
    torch::Tensor translation;
    Points points;

    std::tuple<std::vector<Camera>, Camera *> getCameras(bool validate, const std::string &valImage = "random");

    void saveCameras(const std::string &filename, bool keepCrs);
};
InputData inputDataFromX(const std::string &projectRoot);
std::string findMaskPath(const std::string &imagePath, const std::string &projectRoot);

#endif