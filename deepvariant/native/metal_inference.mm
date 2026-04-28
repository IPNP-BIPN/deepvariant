// MPSGraph implementation of DeepVariant Inception-v3, mirroring
// tools/conversion/inception_v3_mil.py.
//
// All conv+BN pairs are fused on CPU at graph-build time:
//   scale[o]  = 1 / sqrt(var[o] + epsilon)         (gamma is frozen at 1)
//   offset[o] = beta[o] - mean[o] * scale[o]
//   W'[o,i,h,w] = W[o,i,h,w] * scale[o]
// then a single Conv2D + bias-add is emitted to MPSGraph.
//
// MPSGraph data layout: NCHW. We transpose (N,100,221,7) → (N,7,100,221)
// at the input. Concat axis is 1 (channels in NCHW).

#include "deepvariant/native/metal_inference.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphImToColOps.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "deepvariant/native/dv_weights.h"

namespace deepvariant {

namespace {

// Keras `BatchNormalization` defaults to epsilon=1e-3 (NOT 1e-4) and that
// is the value Inception-v3 SavedModels were trained with. Using 1e-4
// produces a subtle scale mismatch on channels where var is small enough
// that the +eps term changes magnitude — large enough to flip the sign
// of post-ReLU activations on those channels, which manifests as
// channel-level mismatch vs TF reference.
constexpr float kBNEpsilon = 1e-3f;

// Build the layer-N variable name used by extract_weights.py.
std::string AttrCpp(int n, const char* attr) {
  return std::string("layer_with_weights-") + std::to_string(n) +
         "/" + attr + "/.ATTRIBUTES/VARIABLE_VALUE";
}

// Fold a Conv (HWIO, FP32) and a BN (gamma=1, beta, mean, var, epsilon)
// into a fused (W', b') pair in HWIO layout (TF-native — no host-side
// transpose; passed straight into MPSGraph with weightsLayout=HWIO).
struct FusedConv {
  std::vector<float> weights_hwio;  // [H, W, I, O]
  std::vector<float> bias;          // [O]
  int O = 0, I = 0, H = 0, W = 0;
};

FusedConv FoldConvBn(const DvwWeights& dvw, int conv_n, int bn_n) {
  const auto* k = dvw.Get(AttrCpp(conv_n, "kernel"));
  const auto* beta = dvw.Get(AttrCpp(bn_n, "beta"));
  const auto* mean = dvw.Get(AttrCpp(bn_n, "moving_mean"));
  const auto* var = dvw.Get(AttrCpp(bn_n, "moving_variance"));
  if (!k || !beta || !mean || !var) {
    LOG(ERROR) << "FoldConvBn(conv=" << conv_n << ", bn=" << bn_n
               << "): missing weight tensor";
    return {};
  }
  if (k->shape.size() != 4u) {
    LOG(ERROR) << "kernel for layer " << conv_n
               << " has rank " << k->shape.size() << " (need 4)";
    return {};
  }
  // Source layout is HWIO (TF Keras convention): shape = (H, W, I, O).
  const int Hk = k->shape[0];
  const int Wk = k->shape[1];
  const int Ik = k->shape[2];
  const int Ok = k->shape[3];
  if (beta->shape.size() != 1u || (int)beta->shape[0] != Ok ||
      mean->shape.size() != 1u || (int)mean->shape[0] != Ok ||
      var->shape.size() != 1u || (int)var->shape[0] != Ok) {
    LOG(ERROR) << "BN params shape mismatch for conv=" << conv_n
               << " bn=" << bn_n;
    return {};
  }
  FusedConv out;
  out.O = Ok;
  out.I = Ik;
  out.H = Hk;
  out.W = Wk;

  // scale[o] = 1 / sqrt(var[o] + eps);  offset[o] = beta[o] - mean[o]*scale[o]
  std::vector<float> scale(Ok), offset(Ok);
  for (int o = 0; o < Ok; ++o) {
    scale[o] = 1.0f / std::sqrt(var->data[o] + kBNEpsilon);
    offset[o] = beta->data[o] - mean->data[o] * scale[o];
  }
  out.bias = std::move(offset);

  // Native HWIO; multiply by scale[o] along the O axis. Optionally
  // flip H and W axes ("true convolution" vs cross-correlation) — see
  // the diagnostic experiment in PORT_LOG. TF uses cross-correlation.
  out.weights_hwio.resize((size_t)Hk * Wk * Ik * Ok);
  const bool flip_spatial = false;  // TF/MPS conv is cross-correlation, no flip
  for (size_t h = 0; h < (size_t)Hk; ++h) {
    const size_t h_src = flip_spatial ? (Hk - 1 - h) : h;
    for (size_t w = 0; w < (size_t)Wk; ++w) {
      const size_t w_src = flip_spatial ? (Wk - 1 - w) : w;
      for (size_t i = 0; i < (size_t)Ik; ++i) {
        for (size_t o = 0; o < (size_t)Ok; ++o) {
          const size_t dst_idx = ((h * Wk + w) * Ik + i) * Ok + o;
          const size_t src_idx = ((h_src * Wk + w_src) * Ik + i) * Ok + o;
          out.weights_hwio[dst_idx] = k->data[src_idx] * scale[o];
        }
      }
    }
  }
  return out;
}

// MPSGraph constant-tensor helper.
//
// IMPORTANT: must use `[[NSData alloc] initWithBytes:length:]` rather
// than `[NSData dataWithBytes:length:]` here. The latter returns an
// AUTORELEASED NSData; when the autoreleasepool from `Create()` drains
// (i.e. before the first `PredictAtTap()` runs), MPSGraph's internal
// reference to the bytes becomes a dangling pointer and the constant
// tensor reads garbage. The +1-retained alloc/init form keeps the
// NSData alive for as long as ARC tracks it through the MPSGraphTensor
// reference graph, surviving past the build-time pool drain.
//
// This is the Phase 5.5a root cause: months of mysterious channel-
// permutation behaviour traced to autoreleased NSData in the conv
// weight constants.
MPSGraphTensor* ConstFloat32(MPSGraph* g, const float* data,
                             NSArray<NSNumber*>* shape, NSString* name) {
  size_t n = 1;
  for (NSNumber* d in shape) n *= [d unsignedIntegerValue];
  NSData* nsdata = [[NSData alloc] initWithBytes:data
                                          length:n * sizeof(float)];
  return [g constantWithData:nsdata
                       shape:shape
                    dataType:MPSDataTypeFloat32];
}

// Build a Conv2D + bias-add via MPSGraph's native
// `convolution2DWithSourceTensor:` (NHWC + HWIO).
//
// Verified bit-exact for the exact stem_s1a shape (input 100×221×7,
// kernel 3×3 stride-2 valid 7→32) by `microtest_metal` (Phase 5.5a
// investigation, Test 6 — known-pattern weights and sparse input,
// hand-computed expected output, max-abs = 0). Earlier reports of a
// channel-permutation bug here were artifacts of an unrelated stale
// shape-mismatch path in `debug_metal`'s TapList — not a real
// MPSGraph issue.
MPSGraphTensor* AddConv(MPSGraph* g, MPSGraphTensor* x,
                        const FusedConv& fc,
                        int stride_y, int stride_x,
                        bool same_padding,  // true = "same", false = "valid"
                        NSString* name) {
  NSArray* w_shape = @[@(fc.H), @(fc.W), @(fc.I), @(fc.O)];
  MPSGraphTensor* W = ConstFloat32(g, fc.weights_hwio.data(), w_shape,
                                   [name stringByAppendingString:@"_w"]);
  MPSGraphTensor* b = ConstFloat32(g, fc.bias.data(), @[@(fc.O)],
                                   [name stringByAppendingString:@"_b"]);

  MPSGraphConvolution2DOpDescriptor* desc =
      [MPSGraphConvolution2DOpDescriptor
          descriptorWithStrideInX:stride_x
                        strideInY:stride_y
                  dilationRateInX:1
                  dilationRateInY:1
                           groups:1
                     paddingStyle:same_padding
                                      ? MPSGraphPaddingStyleTF_SAME
                                      : MPSGraphPaddingStyleTF_VALID
                       dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                    weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
  MPSGraphTensor* y = [g convolution2DWithSourceTensor:x
                                          weightsTensor:W
                                             descriptor:desc
                                                   name:name];
  // Bias broadcast along channel dim. Bias shape (O,) needs reshape to
  // (1, 1, 1, O) for NHWC broadcasting.
  MPSGraphTensor* b_reshaped = [g reshapeTensor:b
                                       withShape:@[@1, @1, @1, @(fc.O)]
                                            name:[name stringByAppendingString:@"_br"]];
  return [g additionWithPrimaryTensor:y
                       secondaryTensor:b_reshaped
                                  name:[name stringByAppendingString:@"_bias"]];
}

// Conv-BN-ReLU: emits the fused conv + bias + relu.
MPSGraphTensor* CBR(MPSGraph* g, MPSGraphTensor* x,
                    const DvwWeights& dvw,
                    int conv_n, int bn_n,
                    int stride_y, int stride_x,
                    bool same_padding,
                    NSString* name) {
  FusedConv fc = FoldConvBn(dvw, conv_n, bn_n);
  if (fc.weights_hwio.empty()) return nullptr;
  MPSGraphTensor* y = AddConv(g, x, fc, stride_y, stride_x, same_padding, name);
  return [g reLUWithTensor:y name:[name stringByAppendingString:@"_r"]];
}

// AvgPool 3×3 + CBR.
MPSGraphTensor* AvgCBR(MPSGraph* g, MPSGraphTensor* x,
                       const DvwWeights& dvw,
                       int conv_n, int bn_n, NSString* name) {
  MPSGraphPooling2DOpDescriptor* pdesc =
      [MPSGraphPooling2DOpDescriptor
          descriptorWithKernelWidth:3
                       kernelHeight:3
                          strideInX:1
                          strideInY:1
                       paddingStyle:MPSGraphPaddingStyleTF_SAME
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC];
  // Keras AvgPool2D / DeepVariant Inception-v3 default is
  // count_include_pad=False (i.e. divide by the number of *real* kernel
  // positions, not by kernel area). MPSGraph defaults to YES, so we
  // override.
  pdesc.includeZeroPadToAverage = NO;
  MPSGraphTensor* p = [g avgPooling2DWithSourceTensor:x
                                            descriptor:pdesc
                                                  name:[name stringByAppendingString:@"_ap"]];
  return CBR(g, p, dvw, conv_n, bn_n, 1, 1, true, name);
}

MPSGraphTensor* MaxPool3x3s2Valid(MPSGraph* g, MPSGraphTensor* x,
                                  NSString* name) {
  MPSGraphPooling2DOpDescriptor* pdesc =
      [MPSGraphPooling2DOpDescriptor
          descriptorWithKernelWidth:3
                       kernelHeight:3
                          strideInX:2
                          strideInY:2
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC];
  pdesc.paddingLeft = 0;
  pdesc.paddingRight = 0;
  pdesc.paddingTop = 0;
  pdesc.paddingBottom = 0;
  return [g maxPooling2DWithSourceTensor:x
                              descriptor:pdesc
                                    name:name];
}

// Inception blocks — direct ports from inception_v3_mil.py.

MPSGraphTensor* Mixed_5b(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=5 (branch1x1)
  MPSGraphTensor* b1 = CBR(g, x, d, 16, 20, 1, 1, true, @"5b_1");
  // M=6 (branch5x5 reduce)
  MPSGraphTensor* b5 = CBR(g, x, d, 12, 14, 1, 1, true, @"5b_5a");
  // M=7 (branch5x5)
  b5 = CBR(g, b5, d, 17, 21, 1, 1, true, @"5b_5b");
  // M=8 (branch3x3dbl reduce)
  MPSGraphTensor* b3 = CBR(g, x, d, 10, 11, 1, 1, true, @"5b_3a");
  // M=9 (branch3x3dbl 3×3)
  b3 = CBR(g, b3, d, 13, 15, 1, 1, true, @"5b_3b");
  // M=10 (branch3x3dbl 3×3)
  b3 = CBR(g, b3, d, 18, 22, 1, 1, true, @"5b_3c");
  // M=11 (branchpool 1×1)
  MPSGraphTensor* bp = AvgCBR(g, x, d, 19, 23, @"5b_p");
  return [g concatTensors:@[b1, b5, b3, bp] dimension:3 name:@"5b"];
}

MPSGraphTensor* Mixed_5c(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=12 (branch1x1)
  MPSGraphTensor* b1 = CBR(g, x, d, 30, 34, 1, 1, true, @"5c_1");
  // M=13 (branch5x5 reduce)
  MPSGraphTensor* b5 = CBR(g, x, d, 26, 28, 1, 1, true, @"5c_5a");
  // M=14 (branch5x5)
  b5 = CBR(g, b5, d, 31, 35, 1, 1, true, @"5c_5b");
  // M=15 (branch3x3dbl reduce)
  MPSGraphTensor* b3 = CBR(g, x, d, 24, 25, 1, 1, true, @"5c_3a");
  // M=16 (branch3x3dbl 3×3)
  b3 = CBR(g, b3, d, 27, 29, 1, 1, true, @"5c_3b");
  // M=17 (branch3x3dbl 3×3)
  b3 = CBR(g, b3, d, 32, 36, 1, 1, true, @"5c_3c");
  // M=18 (branchpool 1×1)
  MPSGraphTensor* bp = AvgCBR(g, x, d, 33, 37, @"5c_p");
  return [g concatTensors:@[b1, b5, b3, bp] dimension:3 name:@"5c"];
}

MPSGraphTensor* Mixed_5d(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=19 (branch1x1)
  MPSGraphTensor* b1 = CBR(g, x, d, 44, 48, 1, 1, true, @"5d_1");
  // M=20 (branch5x5 reduce)
  MPSGraphTensor* b5 = CBR(g, x, d, 40, 42, 1, 1, true, @"5d_5a");
  // M=21 (branch5x5)
  b5 = CBR(g, b5, d, 45, 49, 1, 1, true, @"5d_5b");
  // M=22 (branch3x3dbl reduce)
  MPSGraphTensor* b3 = CBR(g, x, d, 38, 39, 1, 1, true, @"5d_3a");
  // M=23 (branch3x3dbl 3×3)
  b3 = CBR(g, b3, d, 41, 43, 1, 1, true, @"5d_3b");
  // M=24 (branch3x3dbl 3×3)
  b3 = CBR(g, b3, d, 46, 50, 1, 1, true, @"5d_3c");
  // M=25 (branchpool 1×1)
  MPSGraphTensor* bp = AvgCBR(g, x, d, 47, 51, @"5d_p");
  return [g concatTensors:@[b1, b5, b3, bp] dimension:3 name:@"5d"];
}

MPSGraphTensor* Mixed_6a(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=26 (branch3x3 stride 2 valid)
  MPSGraphTensor* b3 = CBR(g, x, d, 56, 58, 2, 2, false, @"6a_3");
  // M=27 (branch3x3dbl_a 1×1)
  MPSGraphTensor* bd = CBR(g, x, d, 52, 53, 1, 1, true, @"6a_da");
  // M=28 (branch3x3dbl_b 3×3)
  bd = CBR(g, bd, d, 54, 55, 1, 1, true, @"6a_db");
  // M=29 (branch3x3dbl_c 3×3 stride 2 valid)
  bd = CBR(g, bd, d, 57, 59, 2, 2, false, @"6a_dc");
  MPSGraphTensor* bp = MaxPool3x3s2Valid(g, x, @"6a_mp");
  return [g concatTensors:@[b3, bd, bp] dimension:3 name:@"6a"];
}

MPSGraphTensor* Mixed_6b(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=30 (b1, 1×1)
  MPSGraphTensor* b1 = CBR(g, x, d, 72, 76, 1, 1, true, @"6b_1");
  // M=31 (b7_a, 1×1 reduce)
  MPSGraphTensor* b7a = CBR(g, x, d, 64, 66, 1, 1, true, @"6b_7aa");
  // M=32 (b7_b, 1×7)
  b7a = CBR(g, b7a, d, 68, 70, 1, 1, true, @"6b_7ab");
  // M=33 (b7_c, 7×1)
  b7a = CBR(g, b7a, d, 73, 77, 1, 1, true, @"6b_7ac");
  // M=34 (b7dbl_a, 1×1 reduce)
  MPSGraphTensor* b7b = CBR(g, x, d, 60, 61, 1, 1, true, @"6b_7ba");
  // M=35 (b7dbl_b, 7×1)
  b7b = CBR(g, b7b, d, 62, 63, 1, 1, true, @"6b_7bb");
  // M=36 (b7dbl_c, 1×7)
  b7b = CBR(g, b7b, d, 65, 67, 1, 1, true, @"6b_7bc");
  // M=37 (b7dbl_d, 7×1)
  b7b = CBR(g, b7b, d, 69, 71, 1, 1, true, @"6b_7bd");
  // M=38 (b7dbl_e, 1×7)
  b7b = CBR(g, b7b, d, 74, 78, 1, 1, true, @"6b_7be");
  // M=39 (bp_1x1)
  MPSGraphTensor* bp = AvgCBR(g, x, d, 75, 79, @"6b_p");
  return [g concatTensors:@[b1, b7a, b7b, bp] dimension:3 name:@"6b"];
}

MPSGraphTensor* Mixed_6c(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=40 (b1, 1×1)
  MPSGraphTensor* b1 = CBR(g, x, d, 92, 96, 1, 1, true, @"6c_1");
  // M=41 (b7_a, 1×1 reduce)
  MPSGraphTensor* b7a = CBR(g, x, d, 84, 86, 1, 1, true, @"6c_7aa");
  // M=42 (b7_b, 1×7)
  b7a = CBR(g, b7a, d, 88, 90, 1, 1, true, @"6c_7ab");
  // M=43 (b7_c, 7×1)
  b7a = CBR(g, b7a, d, 93, 97, 1, 1, true, @"6c_7ac");
  // M=44 (b7dbl_a, 1×1 reduce)
  MPSGraphTensor* b7b = CBR(g, x, d, 80, 81, 1, 1, true, @"6c_7ba");
  // M=45 (b7dbl_b, 7×1)
  b7b = CBR(g, b7b, d, 82, 83, 1, 1, true, @"6c_7bb");
  // M=46 (b7dbl_c, 1×7)
  b7b = CBR(g, b7b, d, 85, 87, 1, 1, true, @"6c_7bc");
  // M=47 (b7dbl_d, 7×1)
  b7b = CBR(g, b7b, d, 89, 91, 1, 1, true, @"6c_7bd");
  // M=48 (b7dbl_e, 1×7)
  b7b = CBR(g, b7b, d, 94, 98, 1, 1, true, @"6c_7be");
  // M=49 (bp_1x1)
  MPSGraphTensor* bp = AvgCBR(g, x, d, 95, 99, @"6c_p");
  return [g concatTensors:@[b1, b7a, b7b, bp] dimension:3 name:@"6c"];
}

MPSGraphTensor* Mixed_6d(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=50 (b1, 1×1)
  MPSGraphTensor* b1 = CBR(g, x, d, 112, 116, 1, 1, true, @"6d_1");
  // M=51 (b7_a, 1×1 reduce)
  MPSGraphTensor* b7a = CBR(g, x, d, 104, 106, 1, 1, true, @"6d_7aa");
  // M=52 (b7_b, 1×7)
  b7a = CBR(g, b7a, d, 108, 110, 1, 1, true, @"6d_7ab");
  // M=53 (b7_c, 7×1)
  b7a = CBR(g, b7a, d, 113, 117, 1, 1, true, @"6d_7ac");
  // M=54 (b7dbl_a, 1×1 reduce)
  MPSGraphTensor* b7b = CBR(g, x, d, 100, 101, 1, 1, true, @"6d_7ba");
  // M=55 (b7dbl_b, 7×1)
  b7b = CBR(g, b7b, d, 102, 103, 1, 1, true, @"6d_7bb");
  // M=56 (b7dbl_c, 1×7)
  b7b = CBR(g, b7b, d, 105, 107, 1, 1, true, @"6d_7bc");
  // M=57 (b7dbl_d, 7×1)
  b7b = CBR(g, b7b, d, 109, 111, 1, 1, true, @"6d_7bd");
  // M=58 (b7dbl_e, 1×7)
  b7b = CBR(g, b7b, d, 114, 118, 1, 1, true, @"6d_7be");
  // M=59 (bp_1x1)
  MPSGraphTensor* bp = AvgCBR(g, x, d, 115, 119, @"6d_p");
  return [g concatTensors:@[b1, b7a, b7b, bp] dimension:3 name:@"6d"];
}

MPSGraphTensor* Mixed_6e(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=60 (b1, 1×1)
  MPSGraphTensor* b1 = CBR(g, x, d, 132, 136, 1, 1, true, @"6e_1");
  // M=61 (b7_a, 1×1 reduce)
  MPSGraphTensor* b7a = CBR(g, x, d, 124, 126, 1, 1, true, @"6e_7aa");
  // M=62 (b7_b, 1×7)
  b7a = CBR(g, b7a, d, 128, 130, 1, 1, true, @"6e_7ab");
  // M=63 (b7_c, 7×1)
  b7a = CBR(g, b7a, d, 133, 137, 1, 1, true, @"6e_7ac");
  // M=64 (b7dbl_a, 1×1 reduce)
  MPSGraphTensor* b7b = CBR(g, x, d, 120, 121, 1, 1, true, @"6e_7ba");
  // M=65 (b7dbl_b, 7×1)
  b7b = CBR(g, b7b, d, 122, 123, 1, 1, true, @"6e_7bb");
  // M=66 (b7dbl_c, 1×7)
  b7b = CBR(g, b7b, d, 125, 127, 1, 1, true, @"6e_7bc");
  // M=67 (b7dbl_d, 7×1)
  b7b = CBR(g, b7b, d, 129, 131, 1, 1, true, @"6e_7bd");
  // M=68 (b7dbl_e, 1×7)
  b7b = CBR(g, b7b, d, 134, 138, 1, 1, true, @"6e_7be");
  // M=69 (bp_1x1)
  MPSGraphTensor* bp = AvgCBR(g, x, d, 135, 139, @"6e_p");
  return [g concatTensors:@[b1, b7a, b7b, bp] dimension:3 name:@"6e"];
}

MPSGraphTensor* Mixed_7a(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=70 (b3_a, 1×1)
  MPSGraphTensor* b3 = CBR(g, x, d, 144, 146, 1, 1, true, @"7a_3a");
  // M=71 (b3_b, 3×3 stride 2 valid)
  b3 = CBR(g, b3, d, 148, 150, 2, 2, false, @"7a_3b");
  // M=72 (b7_a, 1×1)
  MPSGraphTensor* b7 = CBR(g, x, d, 140, 141, 1, 1, true, @"7a_7a");
  // M=73 (b7_b, 1×7)
  b7 = CBR(g, b7, d, 142, 143, 1, 1, true, @"7a_7b");
  // M=74 (b7_c, 7×1)
  b7 = CBR(g, b7, d, 145, 147, 1, 1, true, @"7a_7c");
  // M=75 (b7_d, 3×3 stride 2 valid)
  b7 = CBR(g, b7, d, 149, 151, 2, 2, false, @"7a_7d");
  MPSGraphTensor* bp = MaxPool3x3s2Valid(g, x, @"7a_mp");
  return [g concatTensors:@[b3, b7, bp] dimension:3 name:@"7a"];
}

MPSGraphTensor* Mixed_7b(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=76 (b1, 1×1)
  MPSGraphTensor* b1 = CBR(g, x, d, 162, 168, 1, 1, true, @"7b_1");
  // M=77 (b3 reduce 1×1)
  MPSGraphTensor* b3a = CBR(g, x, d, 154, 156, 1, 1, true, @"7b_3aa");
  // M=78 (b3 1×3)
  MPSGraphTensor* b3a_1x3 = CBR(g, b3a, d, 158, 163, 1, 1, true, @"7b_3a1x3");
  // M=79 (b3 3×1)
  MPSGraphTensor* b3a_3x1 = CBR(g, b3a, d, 159, 164, 1, 1, true, @"7b_3a3x1");
  // M=80 (b3dbl reduce 1×1)
  MPSGraphTensor* b3b = CBR(g, x, d, 152, 153, 1, 1, true, @"7b_3ba");
  // M=81 (b3dbl 3×3)
  b3b = CBR(g, b3b, d, 155, 157, 1, 1, true, @"7b_3bb");
  // M=82 (b3dbl 1×3)
  MPSGraphTensor* b3b_1x3 = CBR(g, b3b, d, 160, 165, 1, 1, true, @"7b_3b1x3");
  // M=83 (b3dbl 3×1)
  MPSGraphTensor* b3b_3x1 = CBR(g, b3b, d, 161, 166, 1, 1, true, @"7b_3b3x1");
  // M=84 (bp 1×1)
  MPSGraphTensor* bp = AvgCBR(g, x, d, 167, 169, @"7b_p");
  return [g concatTensors:@[b1, b3a_1x3, b3a_3x1, b3b_1x3, b3b_3x1, bp] dimension:3 name:@"7b"];
}

MPSGraphTensor* Mixed_7c(MPSGraph* g, MPSGraphTensor* x,
                         const DvwWeights& d) {
  // M=85 (b1, 1×1)
  MPSGraphTensor* b1 = CBR(g, x, d, 180, 186, 1, 1, true, @"7c_1");
  // M=86 (b3 reduce 1×1)
  MPSGraphTensor* b3a = CBR(g, x, d, 172, 174, 1, 1, true, @"7c_3aa");
  // M=87 (b3 1×3)
  MPSGraphTensor* b3a_1x3 = CBR(g, b3a, d, 176, 181, 1, 1, true, @"7c_3a1x3");
  // M=88 (b3 3×1)
  MPSGraphTensor* b3a_3x1 = CBR(g, b3a, d, 177, 182, 1, 1, true, @"7c_3a3x1");
  // M=89 (b3dbl reduce 1×1)
  MPSGraphTensor* b3b = CBR(g, x, d, 170, 171, 1, 1, true, @"7c_3ba");
  // M=90 (b3dbl 3×3)
  b3b = CBR(g, b3b, d, 173, 175, 1, 1, true, @"7c_3bb");
  // M=91 (b3dbl 1×3)
  MPSGraphTensor* b3b_1x3 = CBR(g, b3b, d, 178, 183, 1, 1, true, @"7c_3b1x3");
  // M=92 (b3dbl 3×1)
  MPSGraphTensor* b3b_3x1 = CBR(g, b3b, d, 179, 184, 1, 1, true, @"7c_3b3x1");
  // M=93 (bp 1×1)
  MPSGraphTensor* bp = AvgCBR(g, x, d, 185, 187, @"7c_p");
  return [g concatTensors:@[b1, b3a_1x3, b3a_3x1, b3b_1x3, b3b_3x1, bp] dimension:3 name:@"7c"];
}


}  // namespace

// ---------------------------------------------------------------------------
// Impl: holds device, queue, graph, and the cached executable.
// ---------------------------------------------------------------------------

struct MetalInception::Impl {
  std::unique_ptr<DvwWeights> weights;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  MPSGraph* graph = nil;
  MPSGraphTensor* input = nil;
  MPSGraphTensor* output = nil;
  // Named taps for debugging — keyed by stage name. Populated as the
  // graph is built, so PredictAtTap() can request a specific stage's
  // output.
  NSMutableDictionary<NSString*, MPSGraphTensor*>* taps = nil;
  // Compiled executable with optimizationLevel=Level0 (GPU-only,
  // no ANE placement pass). Lazily filled per tap on first request.
  MPSGraphCompilationDescriptor* compileDesc = nil;
  NSMutableDictionary<NSString*, MPSGraphExecutable*>* execCache = nil;
  // Ordered list of tap tensors compiled into the gap executable
  // (used for the full-network forward — Predict()).
  int feature_dim = 2048;
};

MetalInception::MetalInception() : impl_(std::make_unique<Impl>()) {}
MetalInception::~MetalInception() = default;

int MetalInception::FeatureDim() const {
  return impl_ ? impl_->feature_dim : 0;
}

std::unique_ptr<MetalInception> MetalInception::Create(
    const std::string& dvw_path) {
  auto self = std::unique_ptr<MetalInception>(new MetalInception());
  auto& I = *self->impl_;

  I.weights = DvwWeights::Open(dvw_path);
  if (!I.weights) {
    LOG(ERROR) << "MetalInception::Create: cannot open " << dvw_path;
    return nullptr;
  }

  I.device = MTLCreateSystemDefaultDevice();
  if (!I.device) {
    LOG(ERROR) << "MetalInception::Create: no Metal device available";
    return nullptr;
  }
  I.queue = [I.device newCommandQueue];
  if (!I.queue) {
    LOG(ERROR) << "MetalInception::Create: failed to create command queue";
    return nullptr;
  }

  I.graph = [MPSGraph new];
  // Compilation descriptor: optimizationLevel=Level0 disables the
  // "placement pass dispatching across NeuralEngine and CPU along
  // with the GPU" (per MPSGraph.h). Default Level1 silently picks
  // mixed-precision paths (e.g. FP16 Winograd intermediates) and
  // off-GPU placements for ops where it thinks it's safe — which
  // produces channel-permuted output for our FP32 Inception-v3 conv.
  // Level0 forces GPU-only, full-precision execution at the cost of
  // some perf optimisations.
  I.compileDesc = [MPSGraphCompilationDescriptor new];
  I.compileDesc.optimizationLevel = MPSGraphOptimizationLevel0;
  I.compileDesc.waitForCompilationCompletion = YES;
  I.execCache = [NSMutableDictionary dictionary];
  I.taps = [NSMutableDictionary dictionary];
  // Variable batch dimension. -1 means "any" in MPSGraph shape spec.
  I.input = [I.graph placeholderWithShape:@[@-1, @100, @221, @7]
                                  dataType:MPSDataTypeFloat32
                                      name:@"input_nhwc"];
  // Stay in NHWC throughout — TF native layout. (Earlier OIHW/NCHW path
  // produced channel-permuted output despite a hand-rolled transpose
  // matching TF; switching to NHWC end-to-end resolved it.)
  MPSGraphTensor* x = I.input;
  I.taps[@"input_nchw"] = x;  // tap kept under the old name; layout = NHWC now

  // Stem
  x = CBR(I.graph, x, *I.weights, 0, 1, 2, 2, false, @"s1a");
  if (!x) return nullptr;
  I.taps[@"stem_s1a"] = x;
  x = CBR(I.graph, x, *I.weights, 2, 3, 1, 1, false, @"s2a");
  I.taps[@"stem_s2a"] = x;
  x = CBR(I.graph, x, *I.weights, 4, 5, 1, 1, true,  @"s2b");
  I.taps[@"stem_s2b"] = x;
  x = MaxPool3x3s2Valid(I.graph, x, @"mp3a");
  I.taps[@"stem_mp3a"] = x;
  x = CBR(I.graph, x, *I.weights, 6, 7, 1, 1, false, @"s3b");
  I.taps[@"stem_s3b"] = x;
  x = CBR(I.graph, x, *I.weights, 8, 9, 1, 1, false, @"s4a");
  I.taps[@"stem_s4a"] = x;
  x = MaxPool3x3s2Valid(I.graph, x, @"mp5a");
  I.taps[@"stem_mp5a"] = x;

  // InceptionA
  x = Mixed_5b(I.graph, x, *I.weights); I.taps[@"5b"] = x;
  x = Mixed_5c(I.graph, x, *I.weights); I.taps[@"5c"] = x;
  x = Mixed_5d(I.graph, x, *I.weights); I.taps[@"5d"] = x;
  // Reduction-A
  x = Mixed_6a(I.graph, x, *I.weights); I.taps[@"6a"] = x;
  // InceptionB
  x = Mixed_6b(I.graph, x, *I.weights); I.taps[@"6b"] = x;
  x = Mixed_6c(I.graph, x, *I.weights); I.taps[@"6c"] = x;
  x = Mixed_6d(I.graph, x, *I.weights); I.taps[@"6d"] = x;
  x = Mixed_6e(I.graph, x, *I.weights); I.taps[@"6e"] = x;
  // Reduction-B
  x = Mixed_7a(I.graph, x, *I.weights); I.taps[@"7a"] = x;
  // InceptionC
  x = Mixed_7b(I.graph, x, *I.weights); I.taps[@"7b"] = x;
  x = Mixed_7c(I.graph, x, *I.weights); I.taps[@"7c"] = x;

  // Global avg pool over (H, W) → (N, 2048, 1, 1)
  x = [I.graph meanOfTensor:x axes:@[@1, @2] name:@"gap"];
  // Reshape to (N, 2048)
  x = [I.graph reshapeTensor:x withShape:@[@-1, @2048] name:@"squeeze"];
  I.taps[@"gap"] = x;

  I.output = x;
  return self;
}

bool MetalInception::Predict(const float* input, int batch_size,
                              float* output) {
  int unused = 0;
  return PredictAtTap("gap", input, batch_size, output, &unused);
}

bool MetalInception::PredictAtTap(const std::string& tap_name,
                                   const float* input, int batch_size,
                                   float* output,
                                   int* out_total_elems_per_image) {
  if (!input || !output || batch_size <= 0) {
    LOG(ERROR) << "MetalInception::PredictAtTap: bad args";
    return false;
  }
  auto& I = *impl_;

  @autoreleasepool {
    NSString* tap_ns = [NSString stringWithUTF8String:tap_name.c_str()];
    MPSGraphTensor* tap = I.taps[tap_ns];
    if (!tap) {
      LOG(ERROR) << "MetalInception: unknown tap '" << tap_name << "'";
      return false;
    }

    // Compile (and cache) an executable for this specific tap with
    // optimizationLevel=Level0 — only path that gives correct FP32
    // output (Phase 5.5a investigation).
    MPSGraphExecutable* exe = I.execCache[tap_ns];
    if (!exe) {
      MPSShape* in_shape =
          @[@(batch_size), @100, @221, @7];
      MPSGraphShapedType* in_st =
          [[MPSGraphShapedType alloc] initWithShape:in_shape
                                            dataType:MPSDataTypeFloat32];
      NSDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds_shape =
          @{I.input: in_st};
      exe = [I.graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:I.device]
                                  feeds:feeds_shape
                          targetTensors:@[tap]
                       targetOperations:nil
                  compilationDescriptor:I.compileDesc];
      if (!exe) {
        LOG(ERROR) << "MetalInception::PredictAtTap: compile failed for "
                   << tap_name;
        return false;
      }
      I.execCache[tap_ns] = exe;
    }

    // Wrap input as MPSGraphTensorData.
    const NSUInteger n_in = (NSUInteger)batch_size * 100 * 221 * 7;
    NSData* in_data = [NSData dataWithBytes:input
                                     length:n_in * sizeof(float)];
    MPSGraphTensorData* in_td = [[MPSGraphTensorData alloc]
        initWithDevice:[MPSGraphDevice deviceWithMTLDevice:I.device]
                  data:in_data
                 shape:@[@(batch_size), @100, @221, @7]
              dataType:MPSDataTypeFloat32];

    MPSGraphExecutableExecutionDescriptor* runDesc =
        [MPSGraphExecutableExecutionDescriptor new];
    runDesc.waitUntilCompleted = YES;
    NSArray<MPSGraphTensorData*>* outs =
        [exe runWithMTLCommandQueue:I.queue
                        inputsArray:@[in_td]
                       resultsArray:nil
                executionDescriptor:runDesc];
    if (!outs || outs.count != 1) {
      LOG(ERROR) << "MetalInception::PredictAtTap: run produced "
                 << (outs ? outs.count : 0) << " results (expected 1)";
      return false;
    }
    MPSGraphTensorData* out_td = outs[0];
    NSArray<NSNumber*>* shape = out_td.shape;
    NSUInteger total = 1;
    for (NSNumber* d in shape) total *= [d unsignedIntegerValue];
    if (out_total_elems_per_image && batch_size > 0) {
      *out_total_elems_per_image =
          static_cast<int>(total / (NSUInteger)batch_size);
    }
    [out_td.mpsndarray readBytes:output strideBytes:nil];
  }
  return true;
}

}  // namespace deepvariant
