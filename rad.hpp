#ifndef RAD_H
#define RAD_H

#include <cstddef>
#include <string>
#include <vector>

namespace rad {

// Input splats, laid out exactly as Model::savePly writes them (post keepCrs):
//   means      n*3 world positions
//   featuresDc n*3 SH degree-0 coefficients (not RGB)
//   featuresRest n*K*3, coefficient-major (K in {0, 3, 8, 15})
//   opacities  n   logits
//   scales     n*3 log-space
//   quats      n*4 unnormalized, PLY order rot_0..rot_3 (w, x, y, z)
struct SplatData {
    size_t numPoints = 0;
    size_t numRestCoeffs = 0; // K: 0, 3, 8 or 15 (SH degree 0..3)
    std::vector<float> means;
    std::vector<float> featuresDc;
    std::vector<float> featuresRest;
    std::vector<float> opacities;
    std::vector<float> scales;
    std::vector<float> quats;
};

// Runs the full build-lod pipeline (LoD tree, chunk reordering, encoding)
// and writes the .rad file. Returns false on failure (non-finite input
// values or I/O error), matching build-lod's abort behavior.
bool saveRad(const std::string &filename, const SplatData &data);

}

#endif
