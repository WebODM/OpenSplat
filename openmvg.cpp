#include <filesystem>
#include <nlohmann/json.hpp>
#include "openmvg.hpp"
#include "point_io.hpp"
#include "tensor_math.hpp"
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace torch::indexing;

namespace omvg{

bool read_intrinsics(const json& data, std::unordered_map<uint32_t, Intrinsic> &intrinsics){
    /* Example Intrinsic
        {
            "key": 0,
            "value": {
                "polymorphic_id": 2147483650,
                "polymorphic_name": "pinhole_brown_t2",
                "ptr_wrapper": {
                    "id": 2147483731,
                    "data": {
                        "width": 4000,
                        "height": 3000,
                        "ccdw": 6.16,
                        "focal_length": 2344.1557760362504,
                        "principal_point": [
                            2000.0,
                            1500.0
                        ],
                        "disto_t2": [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0
                        ]
                    }
                }
            }
        }
    */

    for(auto intrin: data["intrinsics"]){
        //std::cout << intrin << std::endl;

        Intrinsic intrinsic;

        intrinsic.projectionType = intrin["value"]["polymorphic_name"];

        std::uint32_t id = intrin["key"].get<uint32_t>();

        intrinsic.width = intrin["value"]["ptr_wrapper"]["data"]["width"].get<uint32_t>();
        intrinsic.height = intrin["value"]["ptr_wrapper"]["data"]["height"].get<uint32_t>();
        
        intrinsic.fx = intrin["value"]["ptr_wrapper"]["data"]["focal_length"].get<float>();
        intrinsic.fy = intrin["value"]["ptr_wrapper"]["data"]["focal_length"].get<float>();

        intrinsic.cx = intrin["value"]["ptr_wrapper"]["data"]["principal_point"][0].get<float>();
        intrinsic.cy = intrin["value"]["ptr_wrapper"]["data"]["principal_point"][1].get<float>();

        // find the key needed
        std::string key;
        for(auto item: intrin["value"]["ptr_wrapper"]["data"].items()){
            std::string key_ = item.key();
            int res = key_.find("dis");
            if(res != std::string::npos){
                key = key_;
                break;
            } 
        }

        int counter = 0;
        for(auto param: intrin["value"]["ptr_wrapper"]["data"][key]){
            switch(counter){
                case 0:{ intrinsic.k1 = param.get<float>(); break;}
                case 1:{ intrinsic.k2 = param.get<float>(); break;}
                case 2:{ intrinsic.k3 = param.get<float>(); break;}
                case 3:{ intrinsic.t1 = param.get<float>(); break;}
                case 4:{ intrinsic.t2 = param.get<float>(); break;}
                default:{break;}
            }
            counter++;
        }


        intrinsics[id] = intrinsic;
    }

    return true;
}

bool read_views(const json& data, std::unordered_map<uint32_t, View> &views){
    /*Example entry for View
        {
            "key": 0,
            "value": {
                "polymorphic_id": 2147483649,
                "polymorphic_name": "view_priors",
                "ptr_wrapper": {
                    "id": 2147483649,
                    "data": {
                        "local_path": "",
                        "filename": "DJI_0001.JPG",
                        "width": 4000,
                        "height": 3000,
                        "id_view": 0,
                        "id_intrinsic": 0,
                        "id_pose": 0,
                        "use_pose_center_prior": true,
                        "center_weight": [
                            5.0,
                            5.0,
                            10.0
                        ],
                        "center": [
                            246834.2629212376,
                            4309982.276090393,
                            1993.892
                        ]
                    }
                }
            }
        }
    */
    for(auto item: data["views"]){
        auto value = item["value"];

        auto image_view = value["ptr_wrapper"]["data"];
        View view;
        view.s_Img_path = image_view["filename"].get<std::string>();

        std::uint32_t id = image_view["id_view"].get<std::uint32_t>();

        view.id_intrinsic = image_view["id_intrinsic"].get<std::uint32_t>();
        view.id_pose = image_view["id_pose"].get<std::uint32_t>();

        view.ui_width = image_view["width"].get<std::uint32_t>();
        view.ui_height = image_view["height"].get<std::uint32_t>();

        views[id] = view;
    }

    return true;
}

bool read_poses(const json& data, std::unordered_map<uint32_t, Pose> &poses){
    /*Example entry for View
        {
            "key": 0,
            "value": {
                "rotation": [
                    [
                        0.9994698938677475,
                        -0.03213424850692021,
                        0.005226980475224173
                    ],
                    [
                        -0.03231189074567745,
                        -0.9987371007397999,
                        0.038472656813614867
                    ],
                    [
                        0.003984089410678775,
                        -0.03862115584435917,
                        -0.9992459824051401
                    ]
                ],
                "center": [
                    246833.93383001933,
                    4309984.296235806,
                    1993.5002170875148
                ]
            }
        }
    */

    /*
        OpenMVG rotation data is stored as columns, must convert to rows
    */

    auto scene_poses = data["extrinsics"];
    for(auto item: scene_poses){
        
        std::uint32_t id = item["key"].get<std::uint32_t>();
        auto value = item["value"];

        Pose pose;
        std::vector<float> omvg_rotation;

        for(auto row: value["rotation"]){
            for(auto r: row){
                omvg_rotation.push_back( r.get<float>() );
            }
        }

        pose.rotation = std::vector<float>(9);

        // convert cols to rows

        pose.rotation[0] = omvg_rotation[0];
        pose.rotation[1] = omvg_rotation[3];
        pose.rotation[2] = omvg_rotation[6];

        pose.rotation[3] = omvg_rotation[1];
        pose.rotation[4] = omvg_rotation[4];
        pose.rotation[5] = omvg_rotation[7];

        pose.rotation[6] = omvg_rotation[2];
        pose.rotation[7] = omvg_rotation[5];
        pose.rotation[8] = omvg_rotation[8];
        

        for(auto c: value["center"]){
            pose.center.push_back( c.get<float>() );
        }

        poses[id] = pose;
    }

    return true;
}

InputData inputDataFromOpenMVG(const std::string &projectRoot){
    InputData ret;
    fs::path cmRoot(projectRoot);
    fs::path reconstructionPath = cmRoot / "sfm_data.json";
    fs::path colorPointCloud = cmRoot / "colorized.ply";

    if (fs::exists(cmRoot / "sfm_data.bin") && !fs::exists(reconstructionPath)){
        throw std::runtime_error("No json found, please use openMVG_main_ConvertSfM_DataFormat with the bin to create the json file");
    }
    if (!fs::exists(cmRoot / "sfm_data.bin") && !fs::exists(reconstructionPath)){
        throw std::runtime_error("No project files found, please check the file path for sfm_data.json or sfm_data.bin");
    }

    if (fs::exists(cmRoot / "cloud_and_poses.ply") && !fs::exists(colorPointCloud)){
        throw std::runtime_error("No colorized.ply found, cloud_and_poses found, please run openMVG_main_ComputeSfM_DataColor and name the output colorized.ply");
    }
    if (!fs::exists(cmRoot / "cloud_and_poses.ply") && !fs::exists(colorPointCloud)){
        throw std::runtime_error("No colorized.ply found, cloud_and_poses.ply found, please run openMVG_main_ComputeSfM_DataColor and name the output colorized.ply");
    }

    std::ifstream f(reconstructionPath.string());
    json data = json::parse(f);
    f.close();

    std::string image_root_path = data["root_path"];

    std::cout << "Images should be in :" << image_root_path << std::endl;
    
    std::unordered_map<uint32_t, Intrinsic> intrinsics;
    std::unordered_map<uint32_t, View> views;
    std::unordered_map<uint32_t, Pose> poses;

    bool intrinsics_ok = read_intrinsics(data, intrinsics);
    bool views_ok = read_views(data, views);
    bool poses_ok = read_poses(data, poses);

    if(!intrinsics_ok){
        std::cerr << "Intrinsics didn't read properly" << std::endl;
    }
    if(!views_ok){
        std::cerr << "Views didn't read properly" << std::endl;
    }
    if(!poses_ok){
        std::cerr << "Poses didn't read properly" << std::endl;
    }

    std::cout << "Found " << intrinsics.size() << " intrinsics" << std::endl;
    std::cout << "Found " << views.size() << " views" << std::endl;
    std::cout << "Found " << poses.size() << " poses" << std::endl;

    // start putting the information into the tensors

    torch::Tensor unorientedPoses = torch::zeros({static_cast<long int>(poses.size()), 4, 4}, torch::kFloat32);
    std::unordered_map<uint32_t,uint32_t> pose_indexes;
    size_t i = 0;
    for (const auto &p : poses){
        std::uint32_t pose_id = p.first;
        Pose pose = p.second;
      
        torch::Tensor R = torch::from_blob(pose.rotation.data(), {static_cast<long>(pose.rotation.size())}, torch::kFloat32);
        R = R.reshape({3, 3});

        torch::Tensor T = torch::from_blob(pose.center.data(), {static_cast<long>(pose.center.size())}, torch::kFloat32);
        T = T.reshape({3, 1});
        
        torch::Tensor Rinv = R.transpose(0, 1);
        torch::Tensor Tinv = torch::matmul(-Rinv, T);

        // because the maps are unordered, need this to keep track of which pose in the tensor is the pose we need
        pose_indexes[pose_id] = i;

        unorientedPoses[i].index_put_({Slice(None, 3), Slice(None, 3)}, R);
        unorientedPoses[i].index_put_({Slice(None, 3), Slice(3, 4)}, Tinv);
        unorientedPoses[i][3][3] = 1.0f;

        // Convert OpenMVG's camera CRS (OpenCV) to OpenGL
        unorientedPoses[i].index_put_({Slice(0, 3), Slice(1,3)}, unorientedPoses[i].index({Slice(0, 3), Slice(1,3)}) * -1.0f);
        i++;
    }

    std::cout << "  " << std::endl;

    auto r = autoScaleAndCenterPoses(unorientedPoses);
    torch::Tensor tposes = std::get<0>(r);
    ret.translation = std::get<1>(r);
    ret.scale = std::get<2>(r);

    for (const auto &item : views){
        std::uint32_t view_id = item.first;
        View v = item.second;

        Intrinsic intrinsic = intrinsics.at(v.id_intrinsic);
        
        if (intrinsic.projectionType != "pinhole" && intrinsic.projectionType != "pinhole_brown_t2"){
            throw std::runtime_error("Camera projection type " + intrinsic.projectionType + " is not supported");
        }

        fs::path thisRoot(image_root_path);
        fs::path image_path =  thisRoot/ v.s_Img_path;

        std::uint32_t current_pose = pose_indexes.at(v.id_pose);

        float normalizer = static_cast<float>((std::max)(intrinsic.width, intrinsic.height));
        ret.cameras.emplace_back(Camera(intrinsic.width, intrinsic.height, 
                            static_cast<float>(intrinsic.fx * normalizer), static_cast<float>(intrinsic.fy * normalizer), 
                            static_cast<float>(static_cast<float>(intrinsic.width) / 2.0f + normalizer * intrinsic.cx), 
                            static_cast<float>(static_cast<float>(intrinsic.height) / 2.0f + normalizer * intrinsic.cy), 
                            static_cast<float>(intrinsic.k1), static_cast<float>(intrinsic.k2), static_cast<float>(intrinsic.k3), 
                            static_cast<float>(intrinsic.t1), static_cast<float>(intrinsic.t2),  
                            
                            tposes[current_pose], image_path.string()));
    }

    PointSet *pSet = readPointSet(colorPointCloud.string());

    torch::Tensor points = pSet->pointsTensor().clone();
    
    ret.points.xyz = (points - ret.translation) * ret.scale;
    ret.points.rgb = pSet->colorsTensor().clone();

    RELEASE_POINTSET(pSet);

    return ret;
}

}
