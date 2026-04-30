// Phase 8 / Tier 6.0 — Deterministic Inception block dispatch.
//
// Each Mixed_X block is encoded as a DetMixedBlock that holds raw
// conv weights + BN params + intermediate MTLBuffers per branch.
// Dispatch fans out branches in parallel onto a single
// MTLCommandBuffer, then concats along the channel axis.
//
// Bypasses MPSGraph entirely → output is bit-deterministic across
// reduction orders (per-thread sequential FMA via MetalConvSerial).
// Combined with the existing det stem chain, replacing all 11 Mixed
// blocks with this dispatch gives 100% deterministic Inception-v3
// inference on Apple GPU.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "deepvariant/native/dv_weights.h"
#include "deepvariant/native/metal_avg_pool.h"
#include "deepvariant/native/metal_bn_relu.h"
#include "deepvariant/native/metal_concat.h"
#include "deepvariant/native/metal_conv_serial.h"

#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLCommandBuffer;
@protocol MTLBuffer;
#endif

namespace deepvariant {

// One conv+BN+ReLU op within a branch. The "raw_buf" holds the conv
// output before BN (since we're using unfolded BN); "out_buf" holds
// the BN+ReLU output that feeds the next op or the concat.
struct DetBranchOp {
#ifdef __OBJC__
  ConvDesc conv;
  id<MTLBuffer> w;          // raw HWIO kernel
  id<MTLBuffer> bias;       // all-zero (unfolded BN: bias absorbed in beta)
  id<MTLBuffer> mean;       // BN moving_mean
  id<MTLBuffer> var;        // BN moving_variance
  id<MTLBuffer> beta;       // BN beta
  id<MTLBuffer> raw_buf;    // post-conv pre-BN scratch
  id<MTLBuffer> out_buf;    // post-BN+ReLU output (chains forward)
#endif
  int out_H = 0, out_W = 0, out_C = 0;
};

// One branch within a Mixed block. May be preceded by an avg-pool
// (the "pool" branch in Mixed_5x/6b-e/7b-c) or a max-pool. Most
// branches are a sequential chain of conv+BN+ReLU ops.
struct DetBranch {
  bool has_avg_pool_pre = false;
  AvgPoolDesc avg_pool{};
#ifdef __OBJC__
  id<MTLBuffer> avg_pool_out;   // input to first op when has_avg_pool_pre
#endif
  std::vector<DetBranchOp> ops;
  // Final branch output — pointer to ops.back().out_buf for plumbing.
};

// One Inception Mixed_X block. Holds N branches that all consume the
// same input buffer; their outputs are concatenated along the channel
// axis to produce the block's output.
struct DetMixedBlock {
  std::string tap_name;             // "5b", "5c", ..., "7c"
  int B = 0;                        // batch size (variable, configured at first dispatch)
  int H_in = 0, W_in = 0, C_in = 0;
  int H_out = 0, W_out = 0, C_out = 0;
  std::vector<DetBranch> branches;
#ifdef __OBJC__
  id<MTLBuffer> concat_out;   // post-concat output (input to next block)
#endif
};

// Build Mixed_5b block layout from .dvw weights + input geometry.
// Allocates all intermediate + output MTLBuffers on `device` for
// batch_size up to `max_B`. Returns a fully-populated DetMixedBlock
// ready for dispatch via DispatchDetMixedBlock().
//
// .dvw weight indices for Mixed_5b (from inception_v3_mil.py audit):
//   branch1   : (conv=16, bn=20)  1×1  192→64
//   branch5_a : (conv=12, bn=14)  1×1  192→48
//   branch5_b : (conv=17, bn=21)  5×5  48→64
//   branch3_a : (conv=10, bn=11)  1×1  192→64
//   branch3_b : (conv=13, bn=15)  3×3  64→96
//   branch3_c : (conv=18, bn=22)  3×3  96→96
//   branchp   : (conv=19, bn=23)  1×1  192→32  (preceded by 3×3 avg-pool)
//
// Returns false on weight-load failure (bundle missing).
#ifdef __OBJC__
bool BuildDetMixed5b(id<MTLDevice> device, const DvwWeights& dvw,
                     int max_B, int H_in, int W_in, int C_in,
                     DetMixedBlock* out_block);
#endif

// Dispatch one Inception block onto `cb`. Reads from `input_buf`,
// writes the concatenated output to `block.concat_out`. The block's
// intermediate buffers are written-and-read internally; caller must
// not access them concurrently.
//
// All four kernels (conv_serial, bn_relu, avg_pool, concat) must be
// non-null and reside on the same MTLDevice as the block buffers.
#ifdef __OBJC__
bool DispatchDetMixedBlock(id<MTLCommandBuffer> cb,
                           MetalConvSerial* conv_serial,
                           MetalBnRelu* bn_relu,
                           MetalAvgPool* avg_pool,
                           MetalConcat* concat,
                           const DetMixedBlock& block,
                           id<MTLBuffer> input_buf,
                           int batch_size);
#endif

}  // namespace deepvariant
