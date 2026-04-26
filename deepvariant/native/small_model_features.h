// Compute the 70 features that upstream's small_model takes as input.
// Feature order (must match upstream make_small_model_examples.py output
// order so the same .mlpackage gives the same predictions):
//
//   0..11   : 12 BaseFeatures
//   12..18  : 7  VariantFeatures
//   19..69  : 51 VAF-context features (offset -25..+25 inclusive)
//
// This is feed-into the small_model.mlpackage from
// tools/conversion/convert_small_model.sh.
#pragma once

#include <cstdint>
#include <vector>

#include "deepvariant/protos/deepvariant.pb.h"

namespace deepvariant {

constexpr int kSmallModelNumFeatures = 70;
constexpr int kSmallModelVafContextWindow = 51;

// Build the 70-feature vector for a candidate, against a chosen subset of
// alt_allele_indices. Returns a vector of length kSmallModelNumFeatures
// in the order the small_model was trained on.
std::vector<float> EncodeSmallModelFeatures(
    const learning::genomics::deepvariant::DeepVariantCall& candidate,
    const std::vector<int>& alt_allele_indices);

}  // namespace deepvariant
