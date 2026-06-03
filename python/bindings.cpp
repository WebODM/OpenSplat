// python/bindings.cpp
//
// pybind11 module exposing OpenSplat's C++ training surface to Python.
// Wraps Model, InputData, Camera, and the input-format loaders. The Python
// package opensplat (python/opensplat/) writes the training loop on top.

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "OpenSplat C++ bindings (internal). Use the opensplat package, not _core, directly.";
}
