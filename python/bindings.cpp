// python/bindings.cpp
//
// pybind11 module exposing OpenSplat's C++ training surface to Python.
// Wraps Model, InputData, Camera, and the input-format loaders. The Python
// package opensplat (python/opensplat/) writes the training loop on top.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <optional>

#include "input_data.hpp"
#include "model.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "OpenSplat C++ bindings (internal). Use the opensplat package, not _core, directly.";

    py::class_<Camera>(m, "Camera")
        .def_readonly("id",        &Camera::id)
        .def_readonly("width",     &Camera::width)
        .def_readonly("height",    &Camera::height)
        .def_readonly("fx",        &Camera::fx)
        .def_readonly("fy",        &Camera::fy)
        .def_readonly("cx",        &Camera::cx)
        .def_readonly("cy",        &Camera::cy)
        .def_readonly("k1",        &Camera::k1)
        .def_readonly("k2",        &Camera::k2)
        .def_readonly("k3",        &Camera::k3)
        .def_readonly("p1",        &Camera::p1)
        .def_readonly("p2",        &Camera::p2)
        .def_readonly("file_path", &Camera::filePath)
        .def_readonly("cam_to_world", &Camera::camToWorld)
        .def("load_image", &Camera::loadImage, py::arg("downscale_factor"),
             py::call_guard<py::gil_scoped_release>())
        .def("get_image",  &Camera::getImage,  py::arg("downscale_factor"),
             py::call_guard<py::gil_scoped_release>());

    py::class_<Points>(m, "Points")
        .def_readonly("xyz", &Points::xyz)
        .def_readonly("rgb", &Points::rgb);

    py::class_<InputData>(m, "InputData")
        .def_readonly("cameras",     &InputData::cameras)
        .def_readonly("scale",       &InputData::scale)
        .def_readonly("translation", &InputData::translation)
        .def_readonly("points",      &InputData::points)
        // Wrap getCameras: the C++ signature returns (vector<Camera>, Camera*),
        // where Camera* is a non-owning pointer into the returned vector (or
        // nullptr when validate=false). Returning the raw pointer to Python with
        // pybind11's default policy would `delete` a Camera the C++ side owns.
        // Convert to optional<Camera> (by value) so ownership is unambiguous.
        .def("get_cameras",
             [](InputData& self, bool validate, const std::string& val_image) {
                 auto [cams, val_ptr] = self.getCameras(validate, val_image);
                 std::optional<Camera> val_cam;
                 if (val_ptr != nullptr) {
                     val_cam = *val_ptr;
                 }
                 return std::make_tuple(std::move(cams), std::move(val_cam));
             },
             py::arg("validate") = false,
             py::arg("val_image") = std::string("random"),
             py::call_guard<py::gil_scoped_release>());

    py::class_<Model>(m, "Model")
        .def(py::init<const InputData&, int, int, int, int, int,
                      int, int, int, float, float, int, float,
                      int, bool, const torch::Device&>(),
             py::arg("input_data"),
             py::arg("num_cameras"),
             py::arg("num_downscales"),
             py::arg("resolution_schedule"),
             py::arg("sh_degree"),
             py::arg("sh_degree_interval"),
             py::arg("refine_every"),
             py::arg("warmup_length"),
             py::arg("reset_alpha_every"),
             py::arg("densify_grad_thresh"),
             py::arg("densify_size_thresh"),
             py::arg("stop_screen_size_at"),
             py::arg("split_screen_size"),
             py::arg("max_steps"),
             py::arg("keep_crs"),
             py::arg("device"),
             py::call_guard<py::gil_scoped_release>())
        .def_readonly("means",         &Model::means)
        .def_readonly("scales",        &Model::scales)
        .def_readonly("quats",         &Model::quats)
        .def_readonly("features_dc",   &Model::featuresDc)
        .def_readonly("features_rest", &Model::featuresRest)
        .def_readonly("opacities",     &Model::opacities)
        .def("forward", &Model::forward,
             py::arg("cam"), py::arg("step"),
             py::call_guard<py::gil_scoped_release>())
        .def("main_loss", &Model::mainLoss,
             py::arg("rendered"), py::arg("gt"), py::arg("ssim_weight"))
        .def("optimizers_zero_grad", &Model::optimizersZeroGrad,
             py::call_guard<py::gil_scoped_release>())
        .def("optimizers_step", &Model::optimizersStep,
             py::call_guard<py::gil_scoped_release>())
        .def("schedulers_step", &Model::schedulersStep, py::arg("step"))
        .def("get_downscale_factor", &Model::getDownscaleFactor, py::arg("step"))
        .def("after_train", &Model::afterTrain, py::arg("step"),
             py::call_guard<py::gil_scoped_release>())
        .def("save", &Model::save,
             py::arg("filename"), py::arg("step"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_ply", &Model::savePly,
             py::arg("filename"), py::arg("step"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_splat", &Model::saveSplat,
             py::arg("filename"),
             py::call_guard<py::gil_scoped_release>())
        .def("load_ply", &Model::loadPly,
             py::arg("filename"),
             py::call_guard<py::gil_scoped_release>());

    m.def("input_data_from_path", &inputDataFromX,
          py::arg("project_root"), py::arg("colmap_image_source_path") = std::string(""),
          py::call_guard<py::gil_scoped_release>(),
          "Detect input format from project_root and load it.");
}
