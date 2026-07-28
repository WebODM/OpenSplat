#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/version.h>
#include "nerfstudio.hpp"
#include "kdtree_tensor.hpp"
#include "spherical_harmonics.hpp"
#include "ssim.hpp"
#include "input_data.hpp"
#include "optim_scheduler.hpp"

using namespace torch::indexing;
using namespace torch::autograd;

torch::Tensor randomQuatTensor(long long n);
torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device);
torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt);
torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt);

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval, 
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs,
        const torch::Device &device) :
    numCameras(numCameras),
    numDownscales(numDownscales), resolutionSchedule(resolutionSchedule), shDegree(shDegree), shDegreeInterval(shDegreeInterval), 
    refineEvery(refineEvery), warmupLength(warmupLength), resetAlphaEvery(resetAlphaEvery), stopSplitAt(maxSteps / 2), densifyGradThresh(densifyGradThresh), densifySizeThresh(densifySizeThresh), stopScreenSizeAt(stopScreenSizeAt), splitScreenSize(splitScreenSize),
    maxSteps(maxSteps), keepCrs(keepCrs),
    device(device), ssim(11, 3){

    long long numPoints = inputData.points.xyz.size(0);
    scale = inputData.scale;
    translation = inputData.translation;

    torch::manual_seed(42);

    means = inputData.points.xyz.to(device).requires_grad_();
    scales = PointsTensor(inputData.points.xyz).scales().repeat({1, 3}).log().to(device).requires_grad_();
    quats = randomQuatTensor(numPoints).to(device).requires_grad_();

    int dimSh = numShBases(shDegree);
    torch::Tensor shs = torch::zeros({numPoints, dimSh, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    shs.index({Slice(), 0, Slice(None, 3)}) = rgb2sh(inputData.points.rgb.toType(torch::kFloat64) / 255.0).toType(torch::kFloat32);
    shs.index({Slice(), Slice(1, None), Slice(3, None)}) = 0.0f;

    featuresDc = shs.index({Slice(), 0, Slice()}).to(device).requires_grad_();
    featuresRest = shs.index({Slice(), Slice(1, None), Slice()}).to(device).requires_grad_();
    opacities = torch::logit(0.1f * torch::ones({numPoints, 1})).to(device).requires_grad_();
    
    backgroundColor = torch::tensor({0.6130f, 0.0101f, 0.3984f}, device).requires_grad_(); // Nerf Studio default

    setupOptimizers();
  }

  ~Model(){
    releaseOptimizers();
  }
  
  void setupOptimizers();
  void releaseOptimizers();

  torch::Tensor forward(Camera& cam, int step);
  void optimizersZeroGrad();
  void optimizersStep();
  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(int step);
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
  void saveSplat(const std::string &filename);
  void saveDebugPly(const std::string &filename, int step);
  int loadPly(const std::string &filename);
  torch::Tensor mainLoss(torch::Tensor &rgb, torch::Tensor &gt, float ssimWeight, const torch::Tensor &mask = torch::Tensor());

  void addToOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &idcs, int nSamples);
  void removeFromOptimizer(torch::optim::Adam *optimizer, const torch::Tensor &newParam, const torch::Tensor &deletedMask);
  torch::Tensor means;
  torch::Tensor scales;
  torch::Tensor quats;
  torch::Tensor featuresDc;
  torch::Tensor featuresRest;
  torch::Tensor opacities;

  torch::optim::Adam *meansOpt = nullptr;
  torch::optim::Adam *scalesOpt = nullptr;
  torch::optim::Adam *quatsOpt = nullptr;
  torch::optim::Adam *featuresDcOpt = nullptr;
  torch::optim::Adam *featuresRestOpt = nullptr;
  torch::optim::Adam *opacitiesOpt = nullptr;

  OptimScheduler *meansOptScheduler = nullptr;

  torch::Tensor radii; // set in forward()
  torch::Tensor xys; // set in forward()
  int lastHeight; // set in forward()
  int lastWidth; // set in forward()

  torch::Tensor xysGradNorm; // set in afterTrain()
  torch::Tensor visCounts; // set in afterTrain()  
  torch::Tensor max2DSize; // set in afterTrain()


  torch::Tensor backgroundColor;
  torch::Device device;
  SSIM ssim;

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
  float densifyGradThresh;
  float densifySizeThresh;
  int stopScreenSizeAt;
  float splitScreenSize;
  int maxSteps;
  bool keepCrs;

  float scale;
  torch::Tensor translation;
};


#endif
