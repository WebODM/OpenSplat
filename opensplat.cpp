#include <filesystem>
#include <nlohmann/json.hpp>
#include "opensplat.hpp"
#include "input_data.hpp"
#include "utils.hpp"
#include "cv_utils.hpp"
#include "constants.hpp"
#include <cxxopts.hpp>

#ifdef USE_VISUALIZATION
#include "visualizer.hpp"
#endif

namespace fs = std::filesystem;
using namespace torch::indexing;

// Drops points from inputData.points that are never seen projecting into a
// masked-in (white) region of any camera that has a mask loaded. Points not
// observed by any masked camera (e.g. behind every frustum) are kept, since
// no camera actually voted to exclude them. This runs once on the initial
// sparse cloud, before Model seeds gaussians from it, so gaussians are only
// ever created where the masks say the subject is.
static void filterPointsByMasks(InputData &inputData){
    torch::Tensor xyz = inputData.points.xyz.to(torch::kCPU).contiguous();
    long long n = xyz.size(0);
    if (n == 0) return;

    std::vector<uint8_t> seen(n, 0);
    std::vector<uint8_t> keep(n, 0);
    auto xyzAcc = xyz.accessor<float, 2>();

    for (Camera &cam : inputData.cameras){
        if (!cam.hasMask()) continue;

        torch::Tensor R = cam.camToWorld.index({Slice(None, 3), Slice(None, 3)});
        torch::Tensor T = cam.camToWorld.index({Slice(None, 3), Slice(3, 4)});
        R = torch::matmul(R, torch::diag(torch::tensor({1.0f, -1.0f, -1.0f})));
        torch::Tensor Rinv = R.transpose(0, 1).contiguous();
        torch::Tensor Tinv = torch::matmul(-Rinv, T).contiguous();

        auto RinvAcc = Rinv.accessor<float, 2>();
        auto TinvAcc = Tinv.accessor<float, 2>();

        torch::Tensor maskT = cam.getMask(1).contiguous();
        auto maskAcc = maskT.accessor<float, 3>();

        for (long long i = 0; i < n; i++){
            float px = xyzAcc[i][0], py = xyzAcc[i][1], pz = xyzAcc[i][2];
            float cx_ = RinvAcc[0][0] * px + RinvAcc[0][1] * py + RinvAcc[0][2] * pz + TinvAcc[0][0];
            float cy_ = RinvAcc[1][0] * px + RinvAcc[1][1] * py + RinvAcc[1][2] * pz + TinvAcc[1][0];
            float cz_ = RinvAcc[2][0] * px + RinvAcc[2][1] * py + RinvAcc[2][2] * pz + TinvAcc[2][0];
            if (cz_ <= 1e-6f) continue;

            int u = static_cast<int>(std::round(cam.fx * cx_ / cz_ + cam.cx));
            int v = static_cast<int>(std::round(cam.fy * cy_ / cz_ + cam.cy));
            if (u < 0 || u >= cam.width || v < 0 || v >= cam.height) continue;

            seen[i] = 1;
            if (maskAcc[v][u][0] >= 0.5f) keep[i] = 1;
        }
    }

    std::vector<int64_t> idxs;
    idxs.reserve(n);
    for (long long i = 0; i < n; i++){
        if (keep[i] || !seen[i]) idxs.push_back(i);
    }

    torch::Tensor idx = torch::tensor(idxs, torch::kInt64);
    inputData.points.xyz = inputData.points.xyz.index_select(0, idx);
    inputData.points.rgb = inputData.points.rgb.index_select(0, idx);
}

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
        ("num-downscales", "Number of images downscales to use. After being scaled by [downscale-factor], images are initially scaled by a further (2^[num-downscales]) and the scale is increased every [resolution-schedule]", cxxopts::value<int>()->default_value("2"))
        ("resolution-schedule", "Double the image resolution every these many steps", cxxopts::value<int>()->default_value("3000"))
        ("sh-degree", "Maximum spherical harmonics degree (must be > 0)", cxxopts::value<int>()->default_value("3"))
        ("sh-degree-interval", "Increase the number of spherical harmonics degree after these many steps (will not exceed [sh-degree])", cxxopts::value<int>()->default_value("1000"))
        ("ssim-weight", "Weight to apply to the structural similarity loss. Set to zero to use least absolute deviation (L1) loss only", cxxopts::value<float>()->default_value("0.2"))
        ("refine-every", "Split/duplicate/prune gaussians every these many steps", cxxopts::value<int>()->default_value("100"))
        ("warmup-length", "Split/duplicate/prune gaussians only after these many steps", cxxopts::value<int>()->default_value("500"))
        ("reset-alpha-every", "Reset the opacity values of gaussians after these many refinements (not steps)", cxxopts::value<int>()->default_value("30"))
        ("densify-grad-thresh", "Threshold of the positional gradient norm (magnitude of the loss function) which when exceeded leads to a gaussian split/duplication", cxxopts::value<float>()->default_value("0.0002"))
        ("densify-size-thresh", "Gaussians' scales below this threshold are duplicated, otherwise split", cxxopts::value<float>()->default_value("0.01"))
        ("stop-screen-size-at", "Stop splitting gaussians that are larger than [split-screen-size] after these many steps", cxxopts::value<int>()->default_value("4000"))
        ("split-screen-size", "Split gaussians that are larger than this percentage of screen space", cxxopts::value<float>()->default_value("0.05"))
        ("colmap-image-path", "Override the default image path for COLMAP-based input", cxxopts::value<std::string>()->default_value(""))
        ("masks-path", "Path to a directory of foreground mask images (one per input image, matched by filename), used to only fit gaussians within the masked-in (white) regions", cxxopts::value<std::string>()->default_value(""))
#ifdef USE_VISUALIZATION
        ("has-visualization", "Show the visualization steps of training", cxxopts::value<bool>()->default_value("0"))
#endif

        ("h,help", "Print usage")
        ("version", "Print version")
        ;
    options.parse_positional({ "input" });
    options.positional_help("[colmap/nerfstudio/opensfm/odm/openmvg project path]");
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
    const int warmupLength = result["warmup-length"].as<int>();
    const int resetAlphaEvery = result["reset-alpha-every"].as<int>();
    const float densifyGradThresh = result["densify-grad-thresh"].as<float>();
    const float densifySizeThresh = result["densify-size-thresh"].as<float>();
    const int stopScreenSizeAt = result["stop-screen-size-at"].as<int>();
    const float splitScreenSize = result["split-screen-size"].as<float>();
    const std::string colmapImageSourcePath = result["colmap-image-path"].as<std::string>();
    const std::string masksPath = result["masks-path"].as<std::string>();
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
        InputData inputData = inputDataFromX(projectRoot, colmapImageSourcePath);

        if (!masksPath.empty()){
            fs::path masksDir(masksPath);
            if (!fs::exists(masksDir)){
                std::cerr << "Masks path does not exist: " << masksPath << std::endl;
                exit(1);
            }

            for (Camera &cam : inputData.cameras){
                fs::path imagePath(cam.filePath);
                for (const std::string &name : { imagePath.filename().string(), imagePath.stem().string() }){
                    for (const std::string &ext : { ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG" }){
                        fs::path maskPath = masksDir / (name + ext);
                        if (fs::exists(maskPath)){
                            cam.maskPath = maskPath.string();
                            break;
                        }
                    }
                    if (!cam.maskPath.empty()) break;
                }
                if (cam.maskPath.empty()){
                    std::cerr << "Warning: no mask found for " << cam.filePath << std::endl;
                }
            }
        }

        parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [&downScaleFactor](Camera &cam){
            cam.loadImage(downScaleFactor);
            cam.loadMask(downScaleFactor);
        });

        if (!masksPath.empty()){
            size_t before = inputData.points.xyz.size(0);
            filterPointsByMasks(inputData);
            std::cout << "Masked point filtering: " << before << " -> " << inputData.points.xyz.size(0) << " points" << std::endl;
        }

        // Withhold a validation camera if necessary
        auto t = inputData.getCameras(validate, valImage);
        std::vector<Camera> cams = std::get<0>(t);
        Camera *valCam = std::get<1>(t);

        Model model(inputData,
                    cams.size(),
                    numDownscales, resolutionSchedule, shDegree, shDegreeInterval, 
                    refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh, densifySizeThresh, stopScreenSizeAt, splitScreenSize,
                    numIters, keepCrs,
                    device);

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

            model.optimizersZeroGrad();

            torch::Tensor rgb = model.forward(cam, step);
            torch::Tensor gt = cam.getImage(model.getDownscaleFactor(step));
            gt = gt.to(device);

            torch::Tensor mask;
            if (cam.hasMask()) mask = cam.getMask(model.getDownscaleFactor(step)).to(device);

            torch::Tensor mainLoss = model.mainLoss(rgb, gt, ssimWeight, mask);
            mainLoss.backward();
            
            if (step % displayStep == 0) {
                const float percentage = static_cast<float>(step) / numIters;
                std::cout << "Step " << step << ": " << mainLoss.item<float>() << " (" << floor(percentage * 100) << "%)" <<  std::endl;
            }

            model.optimizersStep();
            model.schedulersStep(step);
            model.afterTrain(step);

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
            torch::Tensor gt = valCam->getImage(model.getDownscaleFactor(numIters)).to(device);
            torch::Tensor valMask;
            if (valCam->hasMask()) valMask = valCam->getMask(model.getDownscaleFactor(numIters)).to(device);
            std::cout << valCam->filePath << " validation loss: " << model.mainLoss(rgb, gt, ssimWeight, valMask).item<float>() << std::endl;
        }
    }catch(const std::exception &e){
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}
