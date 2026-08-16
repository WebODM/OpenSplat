#include <filesystem>
#include <mutex>
#include <atomic>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime_api.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#include <nlohmann/json.hpp>
#include "input_data.hpp"
#include "cv_utils.hpp"
#include "undistort.hpp"

namespace fs = std::filesystem;
using namespace torch::indexing;
using json = nlohmann::json;

namespace ns{ InputData inputDataFromNerfStudio(const std::string &projectRoot); }
namespace cm{ InputData inputDataFromColmap(const std::string &projectRoot); }
namespace osfm { InputData inputDataFromOpenSfM(const std::string &projectRoot); }
namespace omvg { InputData inputDataFromOpenMVG(const std::string &projectRoot); }

InputData inputDataFromX(const std::string &projectRoot){
    fs::path root(projectRoot);

    if (fs::exists(root / "transforms.json")){
        return ns::inputDataFromNerfStudio(projectRoot);
    }else if (fs::exists(root / "sparse") || fs::exists(root / "cameras.bin")){
        return cm::inputDataFromColmap(projectRoot);
    }else if (fs::exists(root / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM(projectRoot);
    }else if (fs::exists(root / "opensfm" / "reconstruction.json")){
        return osfm::inputDataFromOpenSfM((root / "opensfm").string());
    }else if (fs::exists(root / "sfm_data.json")){
        return omvg::inputDataFromOpenMVG((root).string());
    }
    else{
        throw std::runtime_error("Invalid project folder (must be either a colmap or nerfstudio or openmvg project folder)");
    }
}

torch::Tensor Camera::getIntrinsicsMatrix(){
    return torch::tensor({{fx, 0.0f, cx},
                          {0.0f, fy, cy},
                          {0.0f, 0.0f, 1.0f}}, torch::kFloat32);
}

void Camera::loadImage(float downscaleFactor){
    // Populates image and K, then updates the camera parameters
    // Caution: this function has destructive behaviors
    // and should be called only once
    if (image.numel()) std::runtime_error("loadImage already called");
    
    {
        static std::mutex logMutex;
        std::lock_guard<std::mutex> lock(logMutex);
        std::cout << "Loading " << fs::path(filePath).filename().string() << std::endl;
    }
    
    cv::Mat cImg = imreadRGB(filePath);

    cv::Mat cMask;
    if (!maskPath.empty()){
        cMask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);
        if (cMask.empty()) throw std::runtime_error("Cannot read mask " + maskPath);
    }

    float rescaleF = 1.0f;
    // If camera intrinsics don't match the image dimensions
    if (cImg.rows != height || cImg.cols != width){
        rescaleF = static_cast<float>(cImg.rows) / static_cast<float>(height);
    }
    fx *= rescaleF;
    fy *= rescaleF;
    cx *= rescaleF;
    cy *= rescaleF;

    if (downscaleFactor > 1.0f){
        float scaleFactor = 1.0f / downscaleFactor;
        cv::resize(cImg, cImg, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
        fx *= scaleFactor;
        fy *= scaleFactor;
        cx *= scaleFactor;
        cy *= scaleFactor;
    }

    if (!cMask.empty()){
        if (invertMask) cMask = 255 - cMask;
        cv::threshold(cMask, cMask, 127, 255, cv::THRESH_BINARY);
        if (cMask.rows != cImg.rows || cMask.cols != cImg.cols){
            cv::resize(cMask, cMask, cv::Size(cImg.cols, cImg.rows), 0.0, 0.0, cv::INTER_LINEAR);
        }
    }

    if (hasDistortionParameters()){
        // COLMAP-style undistortion: focal preserved, canvas rescaled
        if (k4 != 0.0f || k5 != 0.0f || k6 != 0.0f){
            std::cout << "Warning: k4/k5/k6 distortion coefficients are ignored" << std::endl;
        }
        UndistortParams p = computeUndistortParams(fx, fy, cx, cy, cImg.cols, cImg.rows,
                                                   k1, k2, k3, p1, p2);
        cv::Mat mapx, mapy;
        buildUndistortMaps(p, mapx, mapy);
        cv::Mat undistorted;
        cv::remap(cImg, undistorted, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        image = imageToTensor(undistorted);
        if (!cMask.empty()){
            cv::Mat remapped;
            cv::remap(cMask, remapped, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            cMask = remapped;
        }
        fx = p.dstFx;
        fy = p.dstFy;
        cx = p.dstCx;
        cy = p.dstCy;
    }else{
        image = imageToTensor(cImg);
    }

    height = image.size(0);
    width = image.size(1);
    K = getIntrinsicsMatrix();

    if (!cMask.empty()){
        torch::Tensor m = torch::from_blob(cMask.data, {cMask.rows, cMask.cols}, torch::kU8)
                            .to(torch::kFloat32).div(255.0f).clone();
        mask = (m >= 0.5f).to(torch::kFloat32);
    }
}

torch::Tensor Camera::getImage(int downscaleFactor){
    if (downscaleFactor <= 1) return image;
    else{

        // torch::jit::script::Module container = torch::jit::load("gt.pt");
        // return container.attr("val").toTensor();

        if (imagePyramids.find(downscaleFactor) != imagePyramids.end()){
            return imagePyramids[downscaleFactor];
        }

        // Rescale, store and return
        cv::Mat cImg = tensorToImage(image);
        cv::resize(cImg, cImg, cv::Size(cImg.cols / downscaleFactor, cImg.rows / downscaleFactor), 0.0, 0.0, cv::INTER_AREA);
        torch::Tensor t = imageToTensor(cImg);
        imagePyramids[downscaleFactor] = t;
        return t;
    }
}

bool Camera::hasDistortionParameters(){
    return k1 != 0.0f || k2 != 0.0f || k3 != 0.0f || k4 != 0.0f || k5 != 0.0f || k6 != 0.0f || p1 != 0.0f || p2 != 0.0f;
}

torch::Tensor Camera::getMask(int downscaleFactor){
    if (!hasMask()) return mask;
    if (downscaleFactor <= 1) return mask;
    if (maskPyramids.find(downscaleFactor) != maskPyramids.end()){
        return maskPyramids[downscaleFactor];
    }
    torch::Tensor m = mask.unsqueeze(0).unsqueeze(0);
    m = torch::nn::functional::interpolate(m,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{ mask.size(0) / downscaleFactor, mask.size(1) / downscaleFactor })
                .mode(torch::kBilinear).align_corners(false));
    m = (m.squeeze(0).squeeze(0) >= 0.5f).to(torch::kFloat32);
    maskPyramids[downscaleFactor] = m;
    return m;
}

bool Camera::gpuCacheEnabled = true;

// Half the free VRAM at first use (CUDA/HIP), a quarter of system RAM on
// Apple unified memory, 1GB otherwise
static long long gpuCacheBudget(){
#ifdef USE_CUDA
    size_t freeB = 0, totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess){
        return static_cast<long long>(freeB / 2);
    }
#elif defined(USE_HIP)
    size_t freeB = 0, totalB = 0;
    if (hipMemGetInfo(&freeB, &totalB) == hipSuccess){
        return static_cast<long long>(freeB / 2);
    }
#endif
#ifdef __APPLE__
    int64_t ram = 0;
    size_t size = sizeof(ram);
    if (sysctlbyname("hw.memsize", &ram, &size, nullptr, 0) == 0){
        return ram / 4;
    }
#endif
    return 1LL << 30;
}

// Cache device-side tensors per camera to avoid re-uploading every iteration.
// Budget-capped; beyond it we fall back to per-iteration uploads.
static torch::Tensor gpuCached(std::unordered_map<int, torch::Tensor> &cache, int key,
                               const torch::Tensor &src, const torch::Device &device){
    if (device == torch::kCPU || !Camera::gpuCacheEnabled) return src.to(device);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    static std::atomic<long long> gpuCacheBytes{0};
    static const long long budget = gpuCacheBudget();
    long long bytes = src.numel() * src.element_size();
    if (gpuCacheBytes.load() + bytes > budget) return src.to(device);
    gpuCacheBytes += bytes;
    torch::Tensor t = src.to(device);
    cache[key] = t;
    return t;
}

torch::Tensor Camera::getImageGpu(int downscaleFactor, const torch::Device &device){
    return gpuCached(gpuImageCache, downscaleFactor, getImage(downscaleFactor), device);
}

torch::Tensor Camera::getMaskGpu(int downscaleFactor, const torch::Device &device){
    torch::Tensor m = getMask(downscaleFactor);
    if (!m.defined() || m.numel() == 0) return m;
    return gpuCached(gpuMaskCache, downscaleFactor, m, device);
}

torch::Tensor Camera::getEdgeMapGpu(int downscaleFactor, const torch::Device &device){
    return gpuCached(gpuEdgeCache, downscaleFactor, getEdgeMap(downscaleFactor).contiguous(), device);
}

torch::Tensor Camera::getEdgeMap(int downscaleFactor){
    if (edgePyramids.find(downscaleFactor) != edgePyramids.end()){
        return edgePyramids[downscaleFactor];
    }
    cv::Mat cImg = tensorToImage(getImage(downscaleFactor));
    cv::Mat gray, edges;
    cv::cvtColor(cImg, gray, cv::COLOR_RGB2GRAY);
    cv::Canny(gray, edges, 50, 150);
    torch::Tensor e = torch::from_blob(edges.data, {edges.rows, edges.cols}, torch::kU8)
                        .to(torch::kFloat32).div(255.0f).clone();
    edgePyramids[downscaleFactor] = e;
    return e;
}

std::string findMaskPath(const std::string &imagePath, const std::string &projectRoot){
    static const char *folders[] = { "masks", "mask", "segmentation", "dynamic_masks" };
    static const char *extensions[] = { ".png", ".jpg", ".jpeg", ".mask.png" };

    fs::path img(imagePath);
    std::string stem = img.stem().string();
    std::string name = img.filename().string();

    for (const char *folder : folders){
        fs::path dir = fs::path(projectRoot) / folder;
        if (!fs::exists(dir) || !fs::is_directory(dir)) continue;
        for (const char *ext : extensions){
            fs::path cand = dir / (stem + ext);
            if (fs::exists(cand)) return cand.string();
            cand = dir / (name + ext);
            if (fs::exists(cand)) return cand.string();
        }
    }
    return "";
}

std::tuple<std::vector<Camera>, Camera *> InputData::getCameras(bool validate, const std::string &valImage){
    if (!validate) return std::make_tuple(cameras, nullptr);
    else{
        size_t valIdx = -1;
        std::srand(42);

        if (valImage == "random"){
            valIdx = std::rand() % cameras.size();
        }else{
            for (size_t i = 0; i < cameras.size(); i++){
                if (fs::path(cameras[i].filePath).filename().string() == valImage){
                    valIdx = i;
                    break;
                }
            }
            if (valIdx == -1) throw std::runtime_error(valImage + " not in the list of cameras");
        }

        std::vector<Camera> cams;
        Camera *valCam = nullptr;

        for (size_t i = 0; i < cameras.size(); i++){
            if (i != valIdx) cams.push_back(cameras[i]);
            else valCam = &cameras[i];
        }

        return std::make_tuple(cams, valCam);
    }
}


void InputData::saveCameras(const std::string &filename, bool keepCrs){
    json j = json::array();
    
    for (size_t i = 0; i < cameras.size(); i++){
        Camera &cam = cameras[i];

        json camera = json::object();
        camera["id"] = i;
        camera["img_name"] = fs::path(cam.filePath).filename().string();
        camera["width"] = cam.width;
        camera["height"] = cam.height;
        camera["fx"] = cam.fx;
        camera["fy"] = cam.fy;

        torch::Tensor R = cam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
        torch::Tensor T = cam.camToWorld.index({Slice(None, 3), Slice(3,4)}).squeeze();
        
        // Flip z and y
        R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f})));

        if (keepCrs) T = (T / scale) + translation;

        std::vector<float> position(3);
        std::vector<std::vector<float>> rotation(3, std::vector<float>(3));
        for (int i = 0; i < 3; i++) {
            position[i] = T[i].item<float>();
            for (int j = 0; j < 3; j++) {
                rotation[i][j] = R[i][j].item<float>();
            }
        }

        camera["position"] = position;
        camera["rotation"] = rotation;
        j.push_back(camera);
    }
    
    std::ofstream of(filename);
    of << j;
    of.close();

    std::cout << "Wrote " << filename << std::endl;
}