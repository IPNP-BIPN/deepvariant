// Phase 8 / Tier 6.0 — Deterministic Inception block dispatch.
//
// Implements BuildDetMixed5b + DispatchDetMixedBlock. Mirrors the
// MPSGraph Mixed_5b build pattern from metal_inference.mm but emits
// Metal kernel chains instead — bit-deterministic across reduction
// orders.

#include "deepvariant/native/metal_det_mixed.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <vector>

#include "absl/log/log.h"

namespace deepvariant {

namespace {

// Keras BatchNormalization default (NOT 1e-4); matches metal_inference.mm
// kBNEpsilon. Must stay in sync.
constexpr float kBNEpsilon = 1e-3f;

// Helper: format DV weight attribute name for layer N. Mirrors
// AttrCpp() in metal_inference.mm. Layer N's kernel = "layer_with_weights-N/kernel/.ATTRIBUTES/VARIABLE_VALUE".
std::string DetAttr(int n, const char* attr) {
  return "layer_with_weights-" + std::to_string(n) +
         "/" + attr + "/.ATTRIBUTES/VARIABLE_VALUE";
}

// Allocate one MTLBuffer of `bytes` size; nil on OOM.
id<MTLBuffer> NewBuf(id<MTLDevice> dev, size_t bytes) {
  return [dev newBufferWithLength:bytes
                          options:MTLResourceStorageModeShared];
}

// Build one CBR (conv + BN + ReLU) op from .dvw weights. Allocates
// raw_buf + out_buf for batch_size up to max_B at the given
// (H_out, W_out, C_out). Returns false on weight-load failure.
bool BuildBranchOp(id<MTLDevice> device, const DvwWeights& dvw,
                   int conv_n, int bn_n,
                   int H_in, int W_in, int C_in,
                   int H_out, int W_out, int stride_h, int stride_w,
                   bool same_padding, int max_B,
                   DetBranchOp* op) {
  const auto* k = dvw.Get(DetAttr(conv_n, "kernel"));
  const auto* beta = dvw.Get(DetAttr(bn_n, "beta"));
  const auto* mean = dvw.Get(DetAttr(bn_n, "moving_mean"));
  const auto* var = dvw.Get(DetAttr(bn_n, "moving_variance"));
  if (!k || !beta || !mean || !var || k->shape.size() != 4u) {
    LOG(ERROR) << "BuildBranchOp: missing weights for conv=" << conv_n
               << " bn=" << bn_n;
    return false;
  }
  const int Hk = k->shape[0], Wk = k->shape[1];
  const int Ik = k->shape[2], Ok = k->shape[3];
  if (Ik != C_in) {
    LOG(ERROR) << "BuildBranchOp: weight C_in=" << Ik
               << " mismatch geometry C_in=" << C_in
               << " (conv=" << conv_n << ")";
    return false;
  }

  op->conv.B = max_B;
  op->conv.H_in = H_in;
  op->conv.W_in = W_in;
  op->conv.C_in = C_in;
  op->conv.H_out = H_out;
  op->conv.W_out = W_out;
  op->conv.C_out = Ok;
  op->conv.Kh = Hk;
  op->conv.Kw = Wk;
  op->conv.stride_h = stride_h;
  op->conv.stride_w = stride_w;
  op->conv.pad_h = same_padding ? (Hk - 1) / 2 : 0;
  op->conv.pad_w = same_padding ? (Wk - 1) / 2 : 0;
  op->conv.relu = false;            // ReLU happens after BN

  op->w = [device newBufferWithBytes:k->data
                              length:k->n_bytes
                             options:MTLResourceStorageModeShared];
  std::vector<float> zero_bias(Ok, 0.0f);
  op->bias = [device newBufferWithBytes:zero_bias.data()
                                 length:Ok * sizeof(float)
                                options:MTLResourceStorageModeShared];
  op->mean = [device newBufferWithBytes:mean->data
                                 length:Ok * sizeof(float)
                                options:MTLResourceStorageModeShared];
  op->var = [device newBufferWithBytes:var->data
                                length:Ok * sizeof(float)
                               options:MTLResourceStorageModeShared];
  op->beta = [device newBufferWithBytes:beta->data
                                 length:Ok * sizeof(float)
                                options:MTLResourceStorageModeShared];
  // Activation buffers — sized for max batch.
  const size_t act_bytes = (size_t)max_B * H_out * W_out * Ok * sizeof(float);
  op->raw_buf = NewBuf(device, act_bytes);
  op->out_buf = NewBuf(device, act_bytes);

  if (!op->w || !op->bias || !op->mean || !op->var || !op->beta ||
      !op->raw_buf || !op->out_buf) {
    LOG(ERROR) << "BuildBranchOp: alloc failed for conv=" << conv_n;
    return false;
  }
  op->out_H = H_out;
  op->out_W = W_out;
  op->out_C = Ok;
  return true;
}

}  // namespace

bool BuildDetMixed5b(id<MTLDevice> device, const DvwWeights& dvw,
                     int max_B, int H_in, int W_in, int C_in,
                     DetMixedBlock* block) {
  if (!device || !block) return false;
  block->tap_name = "5b";
  block->B = max_B;
  block->H_in = H_in;
  block->W_in = W_in;
  block->C_in = C_in;
  // Mixed_5b InceptionA: same padding throughout, no spatial change.
  block->H_out = H_in;
  block->W_out = W_in;
  block->C_out = 64 + 64 + 96 + 32;     // 256 channels post-concat

  block->branches.clear();
  block->branches.resize(4);

  // Branch 0: branch1 — 1×1 conv 192→64.
  {
    DetBranch& br = block->branches[0];
    br.has_avg_pool_pre = false;
    br.ops.resize(1);
    if (!BuildBranchOp(device, dvw, /*conv=*/16, /*bn=*/20,
                       H_in, W_in, C_in, H_in, W_in, 1, 1,
                       /*same=*/true, max_B, &br.ops[0])) return false;
  }

  // Branch 1: branch5 — 1×1 192→48 → 5×5 SAME 48→64.
  {
    DetBranch& br = block->branches[1];
    br.has_avg_pool_pre = false;
    br.ops.resize(2);
    if (!BuildBranchOp(device, dvw, 12, 14,
                       H_in, W_in, C_in, H_in, W_in, 1, 1,
                       true, max_B, &br.ops[0])) return false;
    if (!BuildBranchOp(device, dvw, 17, 21,
                       H_in, W_in, br.ops[0].out_C,
                       H_in, W_in, 1, 1, true, max_B, &br.ops[1])) return false;
  }

  // Branch 2: branch3dbl — 1×1 192→64 → 3×3 SAME 64→96 → 3×3 SAME 96→96.
  {
    DetBranch& br = block->branches[2];
    br.has_avg_pool_pre = false;
    br.ops.resize(3);
    if (!BuildBranchOp(device, dvw, 10, 11,
                       H_in, W_in, C_in, H_in, W_in, 1, 1,
                       true, max_B, &br.ops[0])) return false;
    if (!BuildBranchOp(device, dvw, 13, 15,
                       H_in, W_in, br.ops[0].out_C,
                       H_in, W_in, 1, 1, true, max_B, &br.ops[1])) return false;
    if (!BuildBranchOp(device, dvw, 18, 22,
                       H_in, W_in, br.ops[1].out_C,
                       H_in, W_in, 1, 1, true, max_B, &br.ops[2])) return false;
  }

  // Branch 3: pool — avg-pool 3×3 SAME → 1×1 192→32.
  {
    DetBranch& br = block->branches[3];
    br.has_avg_pool_pre = true;
    br.avg_pool.B = max_B;
    br.avg_pool.H_in = H_in;
    br.avg_pool.W_in = W_in;
    br.avg_pool.C = C_in;
    br.avg_pool.H_out = H_in;
    br.avg_pool.W_out = W_in;
    br.avg_pool.Kh = 3;
    br.avg_pool.Kw = 3;
    br.avg_pool.stride_h = 1;
    br.avg_pool.stride_w = 1;
    br.avg_pool.pad_h = 1;
    br.avg_pool.pad_w = 1;
    br.avg_pool.exclude_pad = true;     // Keras default (count_include_pad=False)
    const size_t pool_bytes =
        (size_t)max_B * H_in * W_in * C_in * sizeof(float);
    br.avg_pool_out = NewBuf(device, pool_bytes);
    if (!br.avg_pool_out) {
      LOG(ERROR) << "BuildDetMixed5b: avg_pool_out alloc failed";
      return false;
    }
    br.ops.resize(1);
    if (!BuildBranchOp(device, dvw, 19, 23,
                       H_in, W_in, C_in, H_in, W_in, 1, 1,
                       true, max_B, &br.ops[0])) return false;
  }

  // Concat output buffer.
  const size_t concat_bytes =
      (size_t)max_B * block->H_out * block->W_out * block->C_out * sizeof(float);
  block->concat_out = NewBuf(device, concat_bytes);
  if (!block->concat_out) {
    LOG(ERROR) << "BuildDetMixed5b: concat_out alloc failed";
    return false;
  }
  return true;
}

bool DispatchDetMixedBlock(id<MTLCommandBuffer> cb,
                           MetalConvSerial* conv_serial,
                           MetalBnRelu* bn_relu,
                           MetalAvgPool* avg_pool,
                           MetalConcat* concat,
                           const DetMixedBlock& block,
                           id<MTLBuffer> input_buf,
                           int batch_size) {
  if (!cb || !conv_serial || !bn_relu || !avg_pool || !concat || !input_buf) {
    LOG(ERROR) << "DispatchDetMixedBlock: nil arg";
    return false;
  }
  if (batch_size <= 0 || batch_size > block.B) {
    LOG(ERROR) << "DispatchDetMixedBlock: batch_size " << batch_size
               << " exceeds capacity " << block.B;
    return false;
  }

  // Run each branch sequentially on the SAME command buffer (Metal will
  // serialize per-encoder; future optimisation: split branches across
  // multiple command buffers for parallelism if profile shows benefit).
  for (size_t bi = 0; bi < block.branches.size(); ++bi) {
    const DetBranch& br = block.branches[bi];
    id<MTLBuffer> branch_in = input_buf;

    if (br.has_avg_pool_pre) {
      AvgPoolDesc apd = br.avg_pool;
      apd.B = batch_size;
      if (!avg_pool->Encode(cb, input_buf, br.avg_pool_out, apd)) {
        LOG(ERROR) << "DispatchDetMixedBlock: avg_pool failed (branch "
                   << bi << ", " << block.tap_name << ")";
        return false;
      }
      branch_in = br.avg_pool_out;
    }

    for (size_t oi = 0; oi < br.ops.size(); ++oi) {
      const DetBranchOp& op = br.ops[oi];
      // 1) Raw conv (no bias, no relu) into raw_buf.
      ConvDesc cd = op.conv;
      cd.B = batch_size;
      if (!conv_serial->Encode(cb, branch_in, op.w, op.bias,
                                op.raw_buf, cd)) {
        LOG(ERROR) << "DispatchDetMixedBlock: conv failed (branch "
                   << bi << " op " << oi << ", " << block.tap_name << ")";
        return false;
      }
      // 2) BN + ReLU into out_buf (consumes raw_buf).
      BnReluDesc bnd{};
      bnd.B = batch_size;
      bnd.H = op.out_H;
      bnd.W = op.out_W;
      bnd.C = op.out_C;
      bnd.eps = kBNEpsilon;
      bnd.relu = true;
      if (!bn_relu->Encode(cb, op.raw_buf, op.mean, op.var, op.beta,
                            op.out_buf, bnd)) {
        LOG(ERROR) << "DispatchDetMixedBlock: bn_relu failed (branch "
                   << bi << " op " << oi << ", " << block.tap_name << ")";
        return false;
      }
      // Chain forward: next op in this branch reads from out_buf.
      branch_in = op.out_buf;
    }
  }

  // Concat all branch outputs along channel axis.
  ConcatDesc ccd{};
  ccd.B = batch_size;
  ccd.H = block.H_out;
  ccd.W = block.W_out;
  ccd.n_branches = static_cast<int>(block.branches.size());
  for (int i = 0; i < 4; ++i) ccd.c_size[i] = 0;
  id<MTLBuffer> br_outs[4] = {nil, nil, nil, nil};
  for (size_t bi = 0; bi < block.branches.size() && bi < 4; ++bi) {
    const DetBranch& br = block.branches[bi];
    if (br.ops.empty()) {
      LOG(ERROR) << "DispatchDetMixedBlock: branch " << bi << " has no ops";
      return false;
    }
    br_outs[bi] = br.ops.back().out_buf;
    ccd.c_size[bi] = br.ops.back().out_C;
  }
  if (!concat->Encode(cb, br_outs[0], br_outs[1], br_outs[2], br_outs[3],
                       block.concat_out, ccd)) {
    LOG(ERROR) << "DispatchDetMixedBlock: concat failed (" << block.tap_name
               << ")";
    return false;
  }
  return true;
}

}  // namespace deepvariant
