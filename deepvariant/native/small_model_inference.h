// Lightweight wrapper around the small_model .mlpackage (3-layer MLP,
// 70 → 750 → 750 → 3). Mirror of CoreMLModel but for vector-shaped input.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deepvariant {

class SmallModel {
 public:
  // Returns nullptr on load failure.
  static std::unique_ptr<SmallModel> Load(const std::string& mlpackage_path);
  ~SmallModel();

  // features: flat vector of N * 70 floats, row-major (one row per candidate).
  // probs: caller-allocated, size N * 3.
  bool Predict(const float* features, int N, float* probs);

  SmallModel(const SmallModel&) = delete;
  SmallModel& operator=(const SmallModel&) = delete;

 private:
  SmallModel();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace deepvariant
