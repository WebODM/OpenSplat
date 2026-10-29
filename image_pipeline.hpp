#ifndef IMAGE_PIPELINE_H
#define IMAGE_PIPELINE_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <torch/torch.h>
#include "image_store.hpp"

// A decoded camera image resident on the training device
struct Frame{
    torch::Tensor image; // uint8 [H,W,3]
    torch::Tensor mask;  // float [H,W] in {0,1}, or undefined
    torch::Tensor edges; // float [H,W] in {0,1}, or undefined
    int imageId = -1;
    int level = 1;
};
// Keeps the frame buffer reserved while the frame is in use
using FrameLease = std::shared_ptr<const Frame>;

class ImagePipeline{
public:
    ImagePipeline(ImageStore &store, const torch::Device &device, int numWorkers);
    ~ImagePipeline();

    void request(const Camera &cam, int level);
    FrameLease acquire(const Camera &cam, int level, bool wantEdges = false);
    void noteTrainStep(double seconds);
    int prefetchTarget() const { return target.load(); }

private:
    struct Slot;
    struct Request{ int imageId; int level; int width; int height; bool hasMask; };

    void workerLoop();
    Slot *reserveSlotLocked(const Request &r);
    void decodeInto(Slot &slot, const Request &r, torch::Tensor &staging, torch::Tensor &maskStaging);
    void retuneLocked(bool waited);

    ImageStore &store;
    torch::Device device;
    std::vector<std::unique_ptr<Slot>> slots;
    std::deque<Request> queue;
    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable queueCv;
    std::condition_variable readyCv;
    bool stopping = false;
    std::string decodeError;

    std::atomic<int> target{2};
    int capacity = 4;
    double decodeEma = 0.0;
    double trainEma = 0.0;   // step time excluding time spent waiting for frames
    double waitAccum = 0.0;  // wait time since the last noteTrainStep
    int windowWaits = 0;     // acquire() calls that had to wait in this window
    int quietWindows = 0;
    int stepsTotal = 0;
    int stepsSinceRetune = 0;
    uint64_t useCounter = 0;
    uint64_t maxSlotBytes = 0;
    uint64_t frameBudget = 0; // device bytes the frame pool may hold, sampled at the first retune
};

#endif
