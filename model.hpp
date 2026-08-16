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
torch::Tensor identityQuatTensor(long long n);
torch::Tensor mrnfKnnLogScales(const torch::Tensor &xyz);
torch::Tensor gumbelTopK(const torch::Tensor &weights, int k);
torch::Tensor projectionMatrix(float zNear, float zFar, float fovX, float fovY, const torch::Device &device);
torch::Tensor psnr(const torch::Tensor& rendered, const torch::Tensor& gt);
torch::Tensor l1(const torch::Tensor& rendered, const torch::Tensor& gt);

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
        int densificationInterval, int densifyFromIter, int densifyUntilIter, int maxGaussians,
        float lossThresh,
        int maxSteps, bool keepCrs,
        const torch::Device &device) :
    numCameras(numCameras),
    numDownscales(numDownscales), resolutionSchedule(resolutionSchedule), shDegree(shDegree), shDegreeInterval(shDegreeInterval),
    densificationInterval(densificationInterval), densifyFromIter(densifyFromIter), densifyUntilIter(densifyUntilIter), maxGaussians(maxGaussians),
    lossThresh(lossThresh),
    maxSteps(maxSteps), keepCrs(keepCrs),
    device(device), ssim(11, 3){

    long long numPoints = inputData.points.xyz.size(0);
    scale = inputData.scale;
    translation = inputData.translation;

    torch::manual_seed(42);

    means = inputData.points.xyz.to(device).requires_grad_();
    scales = PointsTensor(inputData.points.xyz).scales().repeat({1, 3}).log().to(device).requires_grad_();
    quats = identityQuatTensor(numPoints).to(device).requires_grad_();

    int dimSh = numShBases(shDegree);
    torch::Tensor shs = torch::zeros({numPoints, dimSh, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    shs.index({Slice(), 0, Slice(None, 3)}) = rgb2sh(inputData.points.rgb.toType(torch::kFloat64) / 255.0).toType(torch::kFloat32);
    shs.index({Slice(), Slice(1, None), Slice(3, None)}) = 0.0f;

    featuresDc = shs.index({Slice(), 0, Slice()}).to(device).requires_grad_();
    featuresRest = shs.index({Slice(), Slice(1, None), Slice()}).to(device).requires_grad_();
    opacities = torch::logit(0.1f * torch::ones({numPoints, 1})).to(device).requires_grad_();

    backgroundColor = torch::zeros({3}, device);

    // Scene extent from camera positions (vanilla 3DGS cameras_extent)
    spatialLrScale = 1.0f;
    if (!inputData.cameras.empty()){
        torch::Tensor centers = torch::zeros({static_cast<long long>(inputData.cameras.size()), 3});
        for (size_t i = 0; i < inputData.cameras.size(); i++){
            centers[i] = inputData.cameras[i].camToWorld.index({Slice(None, 3), 3});
        }
        torch::Tensor avg = centers.mean(0, true);
        spatialLrScale = (centers - avg).norm(2, 1).max().item<float>() * 1.1f;
        if (spatialLrScale <= 0.0f) spatialLrScale = 1.0f;
    }

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
  void optimizerStepCadence(int step); // FastGS stepping schedule with gradient accumulation
  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  bool afterTrain(int step); // returns true if parameters were restructured
  std::tuple<torch::Tensor, torch::Tensor> computeMultiViewScores(int step, bool densify);
  void densifyAndPrune(int step, const torch::Tensor &importanceScore, const torch::Tensor &pruningScore);
  void resetOpacity(float value);
  void zeroOptimizerRows(torch::optim::Adam *optimizer, const torch::Tensor &idcs);
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
  void saveSplat(const std::string &filename);
  bool saveSpz(const std::string &filename);
  bool saveRad(const std::string &filename);
  void saveDebugPly(const std::string &filename, int step);
  int loadPly(const std::string &filename);
  torch::Tensor mainLoss(torch::Tensor &rgb, torch::Tensor &gt, torch::Tensor &mask, float ssimWeight);

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

  float spatialLrScale = 1.0f;
  std::vector<Camera> *trainCams = nullptr; // set by the trainer, used for multi-view scoring

  torch::Tensor radii; // set in forward()
  torch::Tensor xys; // set in forward()
  torch::Tensor lastAlpha; // set in forward()
  torch::Tensor errorMap; // [H,W] binary metric map for scoring passes, read by rasterize backward
  torch::Tensor densificationInfo; // [4,N] accumulated by rasterize backward
  torch::Tensor xyAbsGrad; // [N,2] Abs-GS screen-gradient accumulation, filled by rasterize backward
  int lastHeight; // set in forward()
  int lastWidth; // set in forward()

  bool scoringPass = false; // true while computeMultiViewScores drives forward/backward
  torch::Tensor xyzGradAccum; // [N] accumulated ||d mean2d||
  torch::Tensor xyzGradAbsAccum; // [N] accumulated ||d mean2d|| (absolute, Abs-GS)
  torch::Tensor gradDenom; // [N] visibility counts
  torch::Tensor maxRadii2D; // [N] max screen radius in px


  torch::Tensor backgroundColor;
  torch::Device device;
  SSIM ssim;

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int densificationInterval;
  int densifyFromIter;
  int densifyUntilIter;
  int maxGaussians;
  float lossThresh;
  int maxSteps;
  bool keepCrs;

  // FastGS hyperparameters (paper defaults)
  float denseThresh = 0.001f;
  float gradThresh = 0.0002f;
  float gradAbsThresh = 0.0012f;
  int opacityResetInterval = 3000;
  int numScoreViews = 10;
  float opacityReg = 0.01f; // global haze/floater penalty on mean opacity
  bool edgeGuidance = true; // Canny-edge weighting of the densification importance

  float scale;
  torch::Tensor translation;
};


#endif
