// python/bindings.cpp
//
// pybind11 module exposing OpenSplat's C++ training surface to Python.
// Wraps Model, InputData, Camera, and the input-format loaders. The Python
// package opensplat (python/opensplat/) writes the training loop on top.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "input_data.hpp"

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
        .def("get_cameras", &InputData::getCameras,
             py::arg("validate"), py::arg("val_image") = std::string("random"),
             py::call_guard<py::gil_scoped_release>());

    m.def("input_data_from_path", &inputDataFromX,
          py::arg("project_root"), py::arg("colmap_image_source_path") = std::string(""),
          py::call_guard<py::gil_scoped_release>(),
          "Detect input format from project_root and load it.");
}
