#include <filesystem>
#include <nlohmann/json.hpp>
#include "opensplat.hpp"
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "constants.hpp"
#include "zip_utils.hpp"
#include <cxxopts.hpp>

#ifdef USE_VISUALIZATION
#include "visualizer.hpp"
#endif

namespace fs = std::filesystem;
using namespace torch::indexing;

int main(int argc, char *argv[]){
    cxxopts::Options options("opensplat", "Open Source 3D Gaussian Splats generator - " APP_VERSION);
    options.add_options()
        ("i,input", "Path to nerfstudio project", cxxopts::value<std::string>())
        ("o,output", "Path where to save output scene", cxxopts::value<std::string>()->default_value("splat.ply"))
        ("s,save-every", "Save output scene every these many steps (set to -1 to disable)", cxxopts::value<int>()->default_value("-1"))
        ("resume", "Resume training from this PLY file", cxxopts::value<std::string>()->default_value(""))
        ("val", "Withhold a camera shot for validating the scene loss")
        ("val-image", "Filename of the image to withhold for validating scene loss", cxxopts::value<std::string>()->default_value("random"))
        ("val-render", "Path of the directory where to render validation images", cxxopts::value<std::string>()->default_value(""))
        ("keep-crs", "Retain the project input's coordinate reference system")
        ("cpu", "Force CPU execution")
        
        ("n,num-iters", "Number of iterations to run", cxxopts::value<int>()->default_value("30000"))
        ("d,downscale-factor", "Scale input images by this factor.", cxxopts::value<float>()->default_value("1"))
        ("num-downscales", "Number of images downscales to use. After being scaled by [downscale-factor], images are initially scaled by a further (2^[num-downscales]) and the scale is increased every [resolution-schedule]", cxxopts::value<int>()->default_value("0"))
        ("resolution-schedule", "Double the image resolution every these many steps", cxxopts::value<int>()->default_value("3000"))
        ("sh-degree", "Maximum spherical harmonics degree (must be > 0)", cxxopts::value<int>()->default_value("3"))
        ("sh-degree-interval", "Increase the number of spherical harmonics degree after these many steps (will not exceed [sh-degree])", cxxopts::value<int>()->default_value("1000"))
        ("ssim-weight", "Weight to apply to the structural similarity loss. Set to zero to use least absolute deviation (L1) loss only", cxxopts::value<float>()->default_value("0.2"))
        ("refine-every", "Densify/prune gaussians every these many steps", cxxopts::value<int>()->default_value("500"))
        ("densify-from", "Start densifying gaussians after these many steps", cxxopts::value<int>()->default_value("500"))
        ("densify-until", "Stop densifying gaussians after these many steps (-1 = min(15000, half of num-iters))", cxxopts::value<int>()->default_value("-1"))
        ("loss-thresh", "High-error pixel threshold on the normalized L1 map for multi-view scoring", cxxopts::value<float>()->default_value("0.1"))
        ("opacity-reg", "Opacity regularization weight (penalizes haze/floaters, 0 to disable)", cxxopts::value<float>()->default_value("0.01"))
        ("max-gaussians", "Maximum number of gaussians (0 = unlimited)", cxxopts::value<int>()->default_value("5000000"))
        ("invert-masks", "Invert image masks (swap object/background)", cxxopts::value<bool>()->default_value("false"))
        ("no-masks", "Ignore image masks even when present", cxxopts::value<bool>()->default_value("false"))
        ("no-gpu-cache", "Do not cache images/masks on the GPU (reduces VRAM usage, slower)", cxxopts::value<bool>()->default_value("false"))
#ifdef USE_VISUALIZATION
        ("has-visualization", "Show the visualization steps of training", cxxopts::value<bool>()->default_value("0"))
#endif
        ("h,help", "Print usage")
        ("version", "Print version")
        ;
    options.parse_positional({ "input" });
    options.positional_help("[colmap/nerfstudio/opensfm/odx/openmvg project path or .zip archive]");
    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("version")){
        std::cout << APP_VERSION << std::endl;
        return EXIT_SUCCESS;
    }
    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }


    const std::string projectRoot = result["input"].as<std::string>();
    const std::string outputScene = result["output"].as<std::string>();
    const int saveEvery = result["save-every"].as<int>(); 
    const std::string resume = result["resume"].as<std::string>();
    const bool validate = result.count("val") > 0 || result.count("val-render") > 0;
    const std::string valImage = result["val-image"].as<std::string>();
    const std::string valRender = result["val-render"].as<std::string>();
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    const bool keepCrs = result.count("keep-crs") > 0;
    const float downScaleFactor = (std::max)(result["downscale-factor"].as<float>(), 1.0f);
    const int numIters = result["num-iters"].as<int>();
    const int numDownscales = result["num-downscales"].as<int>();
    const int resolutionSchedule = result["resolution-schedule"].as<int>();
    const int shDegree = result["sh-degree"].as<int>();
    const int shDegreeInterval = result["sh-degree-interval"].as<int>();
    const float ssimWeight = result["ssim-weight"].as<float>();
    const int refineEvery = result["refine-every"].as<int>();
    const int densifyFrom = result["densify-from"].as<int>();
    int densifyUntil = result["densify-until"].as<int>();
    if (densifyUntil < 0) densifyUntil = (std::min)(15000, result["num-iters"].as<int>() / 2);
    const float lossThresh = result["loss-thresh"].as<float>();
    const int maxGaussians = result["max-gaussians"].as<int>();
    const bool invertMasks = result["invert-masks"].as<bool>();
    const bool noMasks = result["no-masks"].as<bool>();
    #ifdef USE_VISUALIZATION
        const bool hasVisualization = result["has-visualization"].as<bool>();
    #endif

    torch::Device device = torch::kCPU;
    int displayStep = 10;

    if (torch::hasCUDA() && result.count("cpu") == 0) {
        std::cout << "Using CUDA" << std::endl;
        device = torch::kCUDA;
    } else if (torch::hasMPS() && result.count("cpu") == 0) {
        std::cout << "Using MPS" << std::endl;
        device = torch::kMPS;
    }else{
        std::cout << "Using CPU" << std::endl;
        displayStep = 1;
    }

#ifdef USE_VISUALIZATION
    Visualizer visualizer;
    if (hasVisualization)
        visualizer.Initialize(numIters);
#endif

    try{
        std::string projectPath = projectRoot;
        if (isZipArchive(projectRoot)) projectPath = extractZipToCache(projectRoot);
        InputData inputData = inputDataFromX(projectPath);

        int numMasks = 0;
        if (!noMasks){
            for (Camera &cam : inputData.cameras){
                cam.maskPath = findMaskPath(cam.filePath, projectPath);
                cam.invertMask = invertMasks;
                if (!cam.maskPath.empty()) numMasks++;
            }
        }
        if (numMasks > 0) std::cout << "Found " << numMasks << " masks" << std::endl;

        parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [&downScaleFactor](Camera &cam){
            cam.loadImage(downScaleFactor);
        });

        // Withhold a validation camera if necessary
        auto t = inputData.getCameras(validate, valImage);
        std::vector<Camera> cams = std::get<0>(t);
        Camera *valCam = std::get<1>(t);

        Model model(inputData,
                    cams.size(),
                    numDownscales, resolutionSchedule, shDegree, shDegreeInterval,
                    refineEvery, densifyFrom, densifyUntil, maxGaussians,
                    lossThresh,
                    numIters, keepCrs,
                    device);
        model.trainCams = &cams;
        model.opacityReg = result["opacity-reg"].as<float>();
        Camera::gpuCacheEnabled = !result["no-gpu-cache"].as<bool>();

        std::vector< size_t > camIndices( cams.size() );
        std::iota( camIndices.begin(), camIndices.end(), 0 );
        InfiniteRandomIterator<size_t> camsIter( camIndices );

        int imageSize = -1;
        size_t step = 1;

        if (resume != ""){
            step = model.loadPly(resume) + 1;
        }

        for (; step <= numIters; step++){
            Camera& cam = cams[ camsIter.next() ];

            torch::Tensor rgb = model.forward(cam, step);
            torch::Tensor gt = cam.getImageGpu(model.getDownscaleFactor(step), device);
            torch::Tensor mask = cam.getMaskGpu(model.getDownscaleFactor(step), device);

            torch::Tensor mainLoss = model.mainLoss(rgb, gt, mask, ssimWeight);
            mainLoss.backward();
            
            if (step % displayStep == 0) {
                const float percentage = static_cast<float>(step) / numIters;
                std::cout << "Step " << step << ": " << mainLoss.item<float>() << " [" << floor(percentage * 100) << "%]" <<  std::endl;
            }

            model.afterTrain(step);
            model.optimizerStepCadence(step);
            model.schedulersStep(step);

            if (saveEvery > 0 && step % saveEvery == 0){
                fs::path p(outputScene);
                model.save(p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string())).string(), step);
            }

            if (!valRender.empty() && step % 10 == 0){
                torch::Tensor rgb = model.forward(*valCam, step);
                cv::Mat image = tensorToImage(rgb.detach().cpu());
                cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                cv::imwrite((fs::path(valRender) / (std::to_string(step) + ".png")).string(), image);
            }

#ifdef USE_VISUALIZATION
            if (hasVisualization) {
                visualizer.SetInitialGaussianNum(inputData.points.xyz.size(0));
                visualizer.SetLoss(step, mainLoss.item<float>());
                visualizer.SetGaussians(model.means, model.scales, model.featuresDc,
                                        model.opacities);
                visualizer.SetImage(rgb, gt);
                if (visualizer.QuitApp())
                    step = numIters + 1;
                visualizer.Draw();
            }
#endif
        }

        inputData.saveCameras((fs::path(outputScene).parent_path() / "cameras.json").string(), keepCrs);
        model.save(outputScene, numIters);
        // model.saveDebugPly("debug.ply", numIters);

        // Validate
        if (valCam != nullptr){
            torch::Tensor rgb = model.forward(*valCam, numIters);
            torch::Tensor gt = valCam->getImageGpu(model.getDownscaleFactor(numIters), device);
            torch::Tensor valMask = valCam->getMaskGpu(model.getDownscaleFactor(numIters), device);
            std::cout << valCam->filePath << " validation loss: " << model.mainLoss(rgb, gt, valMask, ssimWeight).item<float>() << std::endl;

            torch::Tensor mse;
            if (valMask.defined() && valMask.numel() > 0){
                mse = (valMask.unsqueeze(-1) * (rgb - gt).pow(2)).sum() / (valMask.sum() * gt.size(2) + 1e-8f);
            }else{
                mse = (rgb - gt).pow(2).mean();
            }
            std::cout << valCam->filePath << " validation PSNR: " << (10.0f * torch::log10(1.0f / mse)).item<float>() << std::endl;
        }
    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}
