// MPSGraph + Metal builder for the DeepVariant Inception-v3 big-model
// inference path. Phase 5.5 — replaces coreml_inference for the shipped
// binary. Reads weights from a `.dvw` bundle (see dv_weights.h).
//
// Architecture (mirrors tools/conversion/inception_v3_mil.py):
//
//     input (N, 100, 221, 7) NHWC FP32
//     ↓ NHWC → NCHW transpose
//     ↓ stem: 5× conv-bn-relu + 2× maxpool
//     ↓ 3× InceptionA (Mixed_5b, 5c, 5d)
//     ↓ Reduction-A (Mixed_6a)
//     ↓ 4× InceptionB (Mixed_6b, 6c, 6d, 6e)
//     ↓ Reduction-B (Mixed_7a)
//     ↓ 2× InceptionC (Mixed_7b, 7c)
//     ↓ global avg pool → (N, 2048)
//     output: (N, 2048) FP32 features (pre-dense, pre-softmax)
//
// The final dense (2048→3) + softmax goes through BnnsFinalize for
// deterministic CPU reduction, NOT through this MPSGraph (see
// bnns_finalize.h). That split is what gets us bit-parity with TF on
// the final per-class probabilities.
//
// Threadsafe for Predict() once Create() succeeds; the graph is
// immutable after build.
#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace deepvariant {

class MetalInception {
 public:
  // Open the `.dvw` weight bundle and build the MPSGraph.  Returns
  // nullptr on any error (file missing, weight tensor missing, MPSGraph
  // failure).
  static std::unique_ptr<MetalInception> Create(
      const std::string& dvw_path);

  ~MetalInception();

  // Run inference on a batch of pileup images.
  //
  //   input  : (batch_size, 100, 221, 7) FP32 NHWC, contiguous
  //   output : (batch_size, 2048) FP32 features
  //
  // Returns false on dispatch error.
  bool Predict(const float* input, int batch_size, float* output);

  // Number of feature dimensions in Predict() output (= 2048 for
  // standard WGS Inception-v3).
  int FeatureDim() const;

  MetalInception(const MetalInception&) = delete;
  MetalInception& operator=(const MetalInception&) = delete;

 private:
  MetalInception();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace deepvariant
