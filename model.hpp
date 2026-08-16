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
        int refineEvery, int stopRefine, int growUntil, int maxGaussians,
        int maxSteps, bool keepCrs,
        const torch::Device &device) :
    numCameras(numCameras),
    numDownscales(numDownscales), resolutionSchedule(resolutionSchedule), shDegree(shDegree), shDegreeInterval(shDegreeInterval),
    refineEvery(refineEvery), stopRefine(stopRefine), growUntil(growUntil), maxGaussians(maxGaussians),
    maxSteps(maxSteps), keepCrs(keepCrs),
    device(device), ssim(11, 3){

    long long numPoints = inputData.points.xyz.size(0);
    scale = inputData.scale;
    translation = inputData.translation;

    torch::manual_seed(42);

    means = inputData.points.xyz.to(device).requires_grad_();
    scales = mrnfKnnLogScales(inputData.points.xyz).repeat({1, 3}).to(device).requires_grad_();
    quats = identityQuatTensor(numPoints).to(device).requires_grad_();

    int dimSh = numShBases(shDegree);
    torch::Tensor shs = torch::zeros({numPoints, dimSh, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    shs.index({Slice(), 0, Slice(None, 3)}) = rgb2sh(inputData.points.rgb.toType(torch::kFloat64) / 255.0).toType(torch::kFloat32);
    shs.index({Slice(), Slice(1, None), Slice(3, None)}) = 0.0f;

    featuresDc = shs.index({Slice(), 0, Slice()}).to(device).requires_grad_();
    featuresRest = shs.index({Slice(), Slice(1, None), Slice()}).to(device).requires_grad_();
    opacities = torch::zeros({numPoints, 1}).to(device).requires_grad_(); // logit(0.5)

    backgroundColor = torch::zeros({3}, device); // Black, matching LichtFeld

    setupOptimizers();
    computeBounds();
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
  bool afterTrain(int step); // returns true if parameters were restructured (skip the optimizer step)
  void computeBounds();
  void injectNoise();
  bool refine(int step);
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

  double meanLrUnscaled;
  double scaleLrCurrent;
  double meanLrGamma;
  double scaleLrGamma;

  torch::Tensor radii; // set in forward()
  torch::Tensor xys; // set in forward()
  torch::Tensor lastAlpha; // set in forward()
  torch::Tensor errorMap; // [H,W] densification error map, read by rasterize backward
  torch::Tensor densificationInfo; // [2,N] accumulated by rasterize backward
  int lastHeight; // set in forward()
  int lastWidth; // set in forward()

  torch::Tensor freeMask; // [N] bool, true = dead slot available for reuse
  torch::Tensor visCount; // [N] accumulated blending weights since last refine
  torch::Tensor refineWeightMax; // [N] max over views of per-view error-weighted blending
  torch::Tensor edgeScoreSum; // [N] accumulated edge-weighted blending
  int edgeSampleCount = 0;
  bool edgeGuidance = true;
  float boundsCenter[3];
  float boundsMedianSize = 0.0f;
  float boundsMaxExtent = 0.0f;
  bool boundsValid = false;
  int refinesSinceBounds = 0;


  torch::Tensor backgroundColor;
  torch::Device device;
  SSIM ssim;

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int refineEvery;
  int stopRefine;
  int growUntil;
  int maxGaussians;
  int maxSteps;
  bool keepCrs;

  float scale;
  torch::Tensor translation;
};


#endif
