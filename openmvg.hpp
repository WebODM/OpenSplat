#ifndef OPENMVG_H
#define OPENMVG_H

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <torch/torch.h>
#include <nlohmann/json_fwd.hpp>
#include "input_data.hpp"

using json = nlohmann::json;

namespace omvg{
    struct Intrinsic{
        std::string projectionType = "";

        int width = 0;
        int height = 0;

        float fx = 0;
        float fy = 0;

        float cx = 0;
        float cy = 0;

        float k1 = 0;
        float k2 = 0;
        float k3 = 0;
        float t1 = 0;
        float t2 = 0;
        
    };
    bool read_intrinsics(const json& data, std::unordered_map<uint32_t, Intrinsic> &intrinsics);
    
    struct View{
      // image path on disk
      std::string s_Img_path;

      // Index of intrinsics and the pose
      uint32_t id_intrinsic, id_pose;

      // image size
      uint32_t ui_width, ui_height;
    };
    bool read_views(const json& data, std::unordered_map<uint32_t, View> &views);

    
    struct Pose{
        std::vector<float> rotation;
        std::vector<float> center;
    };
    bool read_poses(const json& data, std::unordered_map<uint32_t, Pose> &poses);

    InputData inputDataFromOpenMVG(const std::string &projectRoot);
}   

#endif