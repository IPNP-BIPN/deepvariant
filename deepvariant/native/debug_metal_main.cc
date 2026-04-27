// Phase 5.5 MPSGraph debug walker.
//
// For an all-zeros input, every Inception-v3 stage in the stem produces
// a spatially-constant per-channel output (since each layer's input is
// spatially constant — first layer = relu(bias), and any conv/pool of
// a spatially constant tensor is also spatially constant). After
// Mixed_5b's branches concatenate, the structure stays spatially
// constant for several more stages.
//
// We use that property to localise the first stage where Metal's
// output goes wrong: at every named tap, we sample the channel 0
// value at multiple spatial positions; if they aren't all equal,
// something has injected spatial structure into the all-constant
// input → that's our divergence point.
//
// Also: for stem_s1a we have a closed-form reference and check
// channel-by-channel exactness (32/32 expected on a healthy build).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "deepvariant/native/dv_weights.h"
#include "deepvariant/native/metal_inference.h"

namespace deepvariant {

const DvwTensor* MustGet(const DvwWeights& w, const std::string& name) {
  const auto* t = w.Get(name);
  if (!t) {
    std::fprintf(stderr, "missing tensor: %s\n", name.c_str());
    std::exit(2);
  }
  return t;
}

// Return (B, C, H, W) inferred from the tap's known geometry.
struct TapShape {
  int C, H, W;
};

const std::vector<std::pair<std::string, TapShape>>& TapList() {
  // Shapes are valid only when the upstream stem is correct; if Metal
  // changes them the dumped buffer size will mismatch.
  static const std::vector<std::pair<std::string, TapShape>> taps = {
      {"stem_s1a",  {32,  49, 110}},
      {"stem_s2a",  {32,  47, 108}},
      {"stem_s2b",  {64,  47, 108}},
      {"stem_mp3a", {64,  23,  53}},
      {"stem_s3b",  {80,  21,  51}},
      {"stem_s4a", {192,  19,  49}},
      {"stem_mp5a",{192,   9,  24}},
      // After the first inception block 5b the spatial dim is 9x24
      // (no spatial change inside inception blocks until reduction).
      {"5b",       {256,   9,  24}},
      {"5c",       {288,   9,  24}},
      {"5d",       {288,   9,  24}},
      {"6a",       {768,   4,  11}},
      {"6b",       {768,   4,  11}},
      {"6c",       {768,   4,  11}},
      {"6d",       {768,   4,  11}},
      {"6e",       {768,   4,  11}},
      {"7a",      {1280,   1,   5}},
      {"7b",      {2048,   1,   5}},
      {"7c",      {2048,   1,   5}},
  };
  return taps;
}

// stem_s1a closed-form check for all-zeros input.
// Returns the per-channel-bias-after-ReLU vector for layer 1 (= the
// spatially-constant value of stem_s1a for an all-zero input).
std::vector<float> CheckStemS1a(const DvwWeights& w, MetalInception& inf) {
  constexpr float kEps = 1e-4f;
  const auto* beta = MustGet(w,
      "layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE");
  const auto* mean = MustGet(w,
      "layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE");
  const auto* var = MustGet(w,
      "layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE");
  const int O = static_cast<int>(beta->shape[0]);

  std::vector<float> expected(O);
  for (int o = 0; o < O; ++o) {
    const float scale = 1.0f / std::sqrt(var->data[o] + kEps);
    expected[o] = std::max(0.0f, beta->data[o] - mean->data[o] * scale);
  }

  constexpr int B = 1, H = 49, W = 110;
  std::vector<float> input((size_t)B * 100 * 221 * 7, 0.0f);
  std::vector<float> output((size_t)B * O * H * W, 0.0f);
  int per = 0;
  inf.PredictAtTap("stem_s1a", input.data(), B, output.data(), &per);

  int n_match = 0;
  for (int o = 0; o < O; ++o) {
    const size_t idx = (((size_t)0 * O + o) * H + H/2) * W + W/2;
    if (output[idx] == expected[o]) ++n_match;
  }
  std::printf("stem_s1a closed-form: %d/%d channels exact\n", n_match, O);
  return expected;
}

// stem_s2a closed-form check for all-zeros input.
// Input to layer 2 is spatially-constant K_in[c] (the layer-1 bias).
// Layer 2 is conv(3x3 stride 1 valid, in=32, out=32), folded with BN.
// At any *interior* pixel (h,w) of the output, value =
//     relu(b_2[o] + sum_c K_in[c] * sum_{dh,dw} W'_2[o, c, dh, dw])
// where W'_2 is the fold-fused kernel (W * scale_2[o]).
void CheckStemS2a(const DvwWeights& w, MetalInception& inf,
                  const std::vector<float>& k_in) {
  constexpr float kEps = 1e-4f;
  const auto* k = MustGet(w,
      "layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE");
  const auto* beta = MustGet(w,
      "layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE");
  const auto* mean = MustGet(w,
      "layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE");
  const auto* var = MustGet(w,
      "layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE");
  // kernel shape (3,3,32,32) HWIO; out_dim=32, in_dim=32.
  const int Hk = 3, Wk = 3;
  const int Ik = (int)k->shape[2];
  const int Ok = (int)k->shape[3];

  std::vector<float> expected(Ok);
  for (int o = 0; o < Ok; ++o) {
    const float scale = 1.0f / std::sqrt(var->data[o] + kEps);
    const float bias_o = beta->data[o] - mean->data[o] * scale;
    // Sum over kernel positions and input channels of W * scale * K_in[c].
    float kernel_sum_times_input = 0.0f;
    for (int i = 0; i < Ik; ++i) {
      float kernel_sum_oi = 0.0f;
      for (int h = 0; h < Hk; ++h) {
        for (int wj = 0; wj < Wk; ++wj) {
          const size_t src = ((size_t)h * Wk + wj) * Ik * Ok +
                             (size_t)i * Ok + o;
          kernel_sum_oi += k->data[src];
        }
      }
      kernel_sum_times_input += k_in[i] * (kernel_sum_oi * scale);
    }
    expected[o] = std::max(0.0f, bias_o + kernel_sum_times_input);
  }

  constexpr int B = 1, H = 47, W = 108;
  std::vector<float> input((size_t)B * 100 * 221 * 7, 0.0f);
  std::vector<float> output((size_t)B * Ok * H * W, 0.0f);
  int per = 0;
  inf.PredictAtTap("stem_s2a", input.data(), B, output.data(), &per);

  // Stem_s2a has VALID padding on stride-1 conv → no spatial variation
  // across all positions for spatially-constant input. Sample center
  // pixel for each channel and compare.
  int n_match = 0, n_close = 0;
  float max_diff = 0.0f;
  for (int o = 0; o < Ok; ++o) {
    const size_t idx = (((size_t)0 * Ok + o) * H + H/2) * W + W/2;
    const float metal_v = output[idx];
    const float diff = std::fabs(metal_v - expected[o]);
    max_diff = std::max(max_diff, diff);
    if (metal_v == expected[o]) ++n_match;
    if (diff < 1e-5f) ++n_close;
  }
  std::printf("stem_s2a closed-form: %d/%d exact, %d/%d <1e-5, max diff %.6e\n",
              n_match, Ok, n_close, Ok, max_diff);
}

// At each tap, sample channel 0 at four corners + center. If the
// values differ, the tensor has spatial structure (which it must NOT
// for a uniform all-zeros input). Print the spatial spread.
void WalkTaps(MetalInception& inf) {
  std::printf("tap          C    H    W   ch0[0,0]      ch0[H/2,W/2]   ch0[H-1,W-1]   spread\n");
  std::printf("-----------  ---  ---  ---  ------------  ------------   ------------   -----------\n");
  for (const auto& [name, sh] : TapList()) {
    const int B = 1;
    const size_t total = (size_t)B * sh.C * sh.H * sh.W;
    std::vector<float> input((size_t)B * 100 * 221 * 7, 0.0f);
    std::vector<float> out(total, 0.0f);
    int per = 0;
    if (!inf.PredictAtTap(name, input.data(), B, out.data(), &per)) {
      std::fprintf(stderr, "tap %s failed\n", name.c_str());
      continue;
    }
    if (per != sh.C * sh.H * sh.W) {
      std::printf("%-11s  shape mismatch: per_image=%d, expected C*H*W=%d\n",
                  name.c_str(), per, sh.C * sh.H * sh.W);
      continue;
    }
    auto at = [&](int c, int h, int w) {
      return out[(((size_t)0 * sh.C + c) * sh.H + h) * sh.W + w];
    };
    const float v00 = at(0, 0, 0);
    const float vmid = at(0, sh.H/2, sh.W/2);
    const float vlast = at(0, sh.H-1, sh.W-1);
    // Spread across all spatial positions of channel 0.
    float vmin = v00, vmax = v00;
    for (int h = 0; h < sh.H; ++h) {
      for (int w = 0; w < sh.W; ++w) {
        const float v = at(0, h, w);
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
      }
    }
    std::printf("%-11s  %3d  %3d  %3d  % .6e  % .6e   % .6e   %.3e\n",
                name.c_str(), sh.C, sh.H, sh.W,
                v00, vmid, vlast, vmax - vmin);
  }
}

int RunDebug(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <wgs.dvw>\n", argv[0]);
    return 2;
  }
  auto w = DvwWeights::Open(argv[1]);
  if (!w) return 1;
  auto inf = MetalInception::Create(argv[1]);
  if (!inf) return 1;

  auto k_layer1 = CheckStemS1a(*w, *inf);
  CheckStemS2a(*w, *inf, k_layer1);
  std::printf("\n");
  WalkTaps(*inf);
  return 0;
}

}  // namespace deepvariant

int main(int argc, char** argv) {
  return deepvariant::RunDebug(argc, argv);
}
