// call_variants — Phase 2 native binary.
//
// Reads a TFRecord of tf.train.Example (pileup images from make_examples),
// runs Inception-v3 inference via Core ML, and writes a TFRecord of
// CallVariantsOutput protos.
//
// Usage:
//   deepvariant call_variants \
//     --examples  /path/make_examples.tfrecord@32 \
//     --checkpoint /path/to/wgs.mlpackage \
//     --outfile    /path/call_variants_output.tfrecord \
//     [--batch_size 128] [--compute_units all|cpu_gpu|cpu_only]
//
// The binary is invoked via the top-level `deepvariant` dispatcher (cli.{h,cc}).

#include "deepvariant/native/call_variants.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#  include <arm_neon.h>
#  define DV_HAVE_NEON 1
#else
#  define DV_HAVE_NEON 0
#endif

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "deepvariant/native/bnns_finalize.h"
#include "deepvariant/native/coreml_inference.h"
#include "deepvariant/native/dv_signpost.h"
#include "deepvariant/native/metal_inference.h"
#include "deepvariant/native/tfrecord.h"
#include "deepvariant/protos/deepvariant.pb.h"
#include "third_party/nucleus/protos/struct.pb.h"
#include "third_party/nucleus/protos/variants.pb.h"
#include "third_party/nucleus/util/utils.h"

ABSL_FLAG(std::string, examples,  "", "Input TFRecord file(s) of tf.train.Example.");
ABSL_FLAG(std::string, checkpoint, "",
          "Inference model path. With --inference_backend=coreml, a "
          ".mlpackage. With --inference_backend=metal, a .dvw weight "
          "bundle (see tools/conversion/extract_weights.py).");
ABSL_FLAG(std::string, outfile,   "", "Output TFRecord file for CallVariantsOutput.");
ABSL_FLAG(int,    batch_size, 128, "Inference batch size.");
ABSL_FLAG(std::string, compute_units, "all",
          "Core ML compute units: all (default), cpu_gpu, cpu_only. "
          "Only applies when --inference_backend=coreml.");
ABSL_FLAG(int, input_height, 100,
          "Pileup-image height for the Metal backend. WGS=100, Trio WGS=140 "
          "(60 child + 2x40 parent), pangenome=100, etc.");
ABSL_FLAG(int, input_channels, 7,
          "Pileup-image channels for the Metal backend. WGS/Trio=7, "
          "pangenome=9.");
ABSL_FLAG(std::string, inference_backend, "metal",
          "Inference backend: metal (default, MPSGraph + BNNS-CPU .dvw — "
          "GPU FP32 on Apple Silicon), coreml (Core ML .mlpackage — ANE "
          "or GPU per --compute_units), or ane_speculate (ANE FP16 first, "
          "GPU FP32 rerun for borderline-confidence sites — Scenario 3 "
          "from the master plan).");
ABSL_FLAG(std::string, ane_speculate_metal_checkpoint, "",
          "When --inference_backend=ane_speculate, the .dvw bundle for "
          "the GPU FP32 rerun on borderline-confidence sites. Required.");
ABSL_FLAG(double, ane_speculate_confidence, 0.99,
          "Borderline threshold for ane_speculate. If max(softmax_ane) < "
          "this value, the example is reclassified on GPU FP32. Lower "
          "→ more GPU reruns, more wall-time, fewer FP-drift artefacts.");

namespace deepvariant {

namespace {

// Parse the tf.train.Example minimal proto to extract features.
// We only do minimal wire-level parsing; see tools/conversion/bench.py for
// the Python equivalent.
struct ExampleFeatures {
  std::string image_encoded;   // bytes_list value of "image/encoded"
  std::string variant_encoded; // bytes_list value of "variant/encoded"
  std::string alt_allele_indices_encoded; // "alt_allele_indices/encoded"
};

// Read a varint from buf starting at position i. Returns (value, new_i).
static uint64_t ReadVarint(const uint8_t* buf, size_t len, size_t& i) {
  uint64_t val = 0;
  int shift = 0;
  while (i < len) {
    uint8_t b = buf[i++];
    val |= static_cast<uint64_t>(b & 0x7F) << shift;
    if (!(b & 0x80)) return val;
    shift += 7;
  }
  return val;  // truncated
}

// Extract a single bytes value from a BytesList field (wire type 2).
// Extracts the first bytes-value from a Feature whose payload is a BytesList.
// The input is the raw bytes of a Feature proto (the value side of a
// map<string, Feature> entry). The Feature is a oneof — field 1 is BytesList.
// BytesList itself has `repeated bytes value = 1;` — each value is a
// length-delimited bytes entry. We walk both levels and return the first
// value's raw bytes (with no proto framing).
static std::string ExtractBytesListFirst(const uint8_t* buf, size_t len) {
  size_t i = 0;
  while (i < len) {
    uint64_t tag = ReadVarint(buf, len, i);
    uint32_t field = static_cast<uint32_t>(tag >> 3);
    uint32_t wire  = static_cast<uint32_t>(tag & 7);
    if (wire != 2) break;  // we only handle length-delimited
    uint64_t seg_len = ReadVarint(buf, len, i);
    if (i + seg_len > len) break;
    if (field == 1) {
      // We're inside Feature.bytes_list — recurse one level to read the
      // first BytesList.value entry (also a length-delimited bytes field).
      const uint8_t* inner = buf + i;
      size_t j = 0;
      while (j < seg_len) {
        uint64_t itag = ReadVarint(inner, seg_len, j);
        uint32_t ifield = static_cast<uint32_t>(itag >> 3);
        uint32_t iwire  = static_cast<uint32_t>(itag & 7);
        if (iwire != 2) break;
        uint64_t ilen = ReadVarint(inner, seg_len, j);
        if (j + ilen > seg_len) break;
        if (ifield == 1) {
          return std::string(reinterpret_cast<const char*>(inner + j), ilen);
        }
        j += ilen;
      }
      return {};
    }
    i += seg_len;
  }
  return {};
}

// Parse a tf.train.Example wire to extract key fields.
// tf.train.Example has one field: features (field=1, wire=2) → Features
// Features has one repeated field: feature (field=1, wire=2) → map<string, Feature>
// Each map entry: key (field=1), value (field=2).
// Feature is a oneof: bytes_list (field=1), float_list (field=2), int64_list (field=3).
static ExampleFeatures ParseExample(const std::string& payload) {
  ExampleFeatures out;
  const uint8_t* buf = reinterpret_cast<const uint8_t*>(payload.data());
  size_t n = payload.size();
  size_t i = 0;

  // Walk top-level Example proto.
  while (i < n) {
    uint64_t tag = ReadVarint(buf, n, i);
    uint32_t wire = tag & 7;
    if (wire != 2) { break; }
    uint64_t seg_len = ReadVarint(buf, n, i);
    if (i + seg_len > n) break;
    // field 1 = Features
    // Walk the Features proto.
    const uint8_t* feat_buf = buf + i;
    size_t feat_len = seg_len;
    i += seg_len;

    size_t fi = 0;
    while (fi < feat_len) {
      uint64_t ftag = ReadVarint(feat_buf, feat_len, fi);
      uint32_t fwire = ftag & 7;
      if (fwire != 2) break;
      uint64_t entry_len = ReadVarint(feat_buf, feat_len, fi);
      if (fi + entry_len > feat_len) break;
      const uint8_t* entry = feat_buf + fi;
      fi += entry_len;

      // Parse map entry: key (field=1), value (field=2).
      std::string key;
      std::string value_bytes;
      size_t ei = 0;
      while (ei < entry_len) {
        uint64_t etag = ReadVarint(entry, entry_len, ei);
        uint32_t ewire = etag & 7;
        uint32_t efd   = etag >> 3;
        if (ewire != 2) { break; }
        uint64_t elen = ReadVarint(entry, entry_len, ei);
        if (ei + elen > entry_len) break;
        if (efd == 1) {
          key.assign(reinterpret_cast<const char*>(entry + ei), elen);
        } else if (efd == 2) {
          // Feature oneof; field=1 = BytesList
          value_bytes.assign(reinterpret_cast<const char*>(entry + ei), elen);
        }
        ei += elen;
      }

      if (key == "image/encoded" || key == "image") {
        // BytesList → first value
        out.image_encoded = ExtractBytesListFirst(
            reinterpret_cast<const uint8_t*>(value_bytes.data()),
            value_bytes.size());
      } else if (key == "variant/encoded") {
        out.variant_encoded = ExtractBytesListFirst(
            reinterpret_cast<const uint8_t*>(value_bytes.data()),
            value_bytes.size());
      } else if (key == "alt_allele_indices/encoded") {
        out.alt_allele_indices_encoded = ExtractBytesListFirst(
            reinterpret_cast<const uint8_t*>(value_bytes.data()),
            value_bytes.size());
      }
    }
  }
  return out;
}

ComputeUnits ParseComputeUnits(const std::string& s) {
  if (s == "cpu_gpu")  return ComputeUnits::kCpuAndGpu;
  if (s == "cpu_only") return ComputeUnits::kCpuOnly;
  return ComputeUnits::kAll;
}

}  // namespace

int RunCallVariants(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string examples_path  = absl::GetFlag(FLAGS_examples);
  const std::string checkpoint_path = absl::GetFlag(FLAGS_checkpoint);
  const std::string outfile_path   = absl::GetFlag(FLAGS_outfile);
  const int batch_size             = absl::GetFlag(FLAGS_batch_size);
  const ComputeUnits compute_units =
      ParseComputeUnits(absl::GetFlag(FLAGS_compute_units));

  if (examples_path.empty() || checkpoint_path.empty() || outfile_path.empty()) {
    LOG(ERROR) << "Required flags: --examples, --checkpoint, --outfile";
    return 2;
  }

  // Pick inference backend.
  const std::string backend = absl::GetFlag(FLAGS_inference_backend);
  std::unique_ptr<CoreMLModel> coreml_model;
  std::unique_ptr<MetalInception> metal_model;
  std::unique_ptr<BnnsFinalize> metal_finalize;
  int H = 0, W = 0, C = 0, K = 0;
  if (backend == "coreml") {
    LOG(INFO) << "Loading Core ML model: " << checkpoint_path;
    coreml_model = CoreMLModel::Load(checkpoint_path, compute_units);
    if (!coreml_model) {
      LOG(ERROR) << "Failed to load Core ML model: " << checkpoint_path;
      return 1;
    }
    H = coreml_model->InputHeight();
    W = coreml_model->InputWidth();
    C = coreml_model->InputChannels();
    K = coreml_model->NumClasses();
  } else if (backend == "metal") {
    LOG(INFO) << "Loading Metal/BNNS model: " << checkpoint_path;
    // Pass --input_height / --input_channels to MetalInception so the
    // MPSGraph placeholder is built with the right shape. Defaults
    // (100×221×7) match WGS; trio passes 140 via --input_height.
    H = absl::GetFlag(FLAGS_input_height);
    W = 221;
    C = absl::GetFlag(FLAGS_input_channels);
    K = 3;
    metal_model = MetalInception::Create(checkpoint_path, H, C);
    metal_finalize = BnnsFinalize::Create(checkpoint_path);
    if (!metal_model || !metal_finalize) {
      LOG(ERROR) << "Failed to load Metal/BNNS model: " << checkpoint_path;
      return 1;
    }
  } else if (backend == "ane_speculate") {
    // Scenario 3: ANE FP16 forward pass on every example; for examples
    // where max(softmax_ane) < threshold (= --ane_speculate_confidence,
    // default 0.99), rerun on GPU MPSGraph FP32 + BNNS-CPU finalize so
    // borderline GQ=20 sites stay on the deterministic FP32 path.
    const std::string metal_ckpt =
        absl::GetFlag(FLAGS_ane_speculate_metal_checkpoint);
    if (metal_ckpt.empty()) {
      LOG(ERROR) << "ane_speculate requires --ane_speculate_metal_checkpoint=<.dvw>";
      return 2;
    }
    LOG(INFO) << "Loading ane_speculate ANE model:   " << checkpoint_path;
    coreml_model = CoreMLModel::Load(checkpoint_path, compute_units);
    if (!coreml_model) {
      LOG(ERROR) << "Failed to load Core ML .mlpackage: " << checkpoint_path;
      return 1;
    }
    LOG(INFO) << "Loading ane_speculate GPU rerun:   " << metal_ckpt;
    H = absl::GetFlag(FLAGS_input_height);
    W = 221;
    C = absl::GetFlag(FLAGS_input_channels);
    K = 3;
    metal_model = MetalInception::Create(metal_ckpt, H, C);
    metal_finalize = BnnsFinalize::Create(metal_ckpt);
    if (!metal_model || !metal_finalize) {
      LOG(ERROR) << "Failed to load .dvw fallback bundle: " << metal_ckpt;
      return 1;
    }
    // Sanity: ANE model and Metal model must agree on input shape.
    if (coreml_model->InputHeight() != H || coreml_model->InputChannels() != C) {
      LOG(ERROR) << "ane_speculate: shape mismatch — ANE expects ("
                 << coreml_model->InputHeight() << "x" << coreml_model->InputWidth()
                 << "x" << coreml_model->InputChannels()
                 << ") but Metal model wants (" << H << "x" << W << "x" << C << ")";
      return 1;
    }
  } else {
    LOG(ERROR) << "Unknown --inference_backend=" << backend
               << " (expected 'coreml', 'metal' or 'ane_speculate')";
    return 2;
  }
  LOG(INFO) << "Model input (" << H << "," << W << "," << C
            << ") → " << K << " classes  [backend=" << backend << "]";

  // Open TFRecord reader + writer.
  auto reader = TFRecordReader::New(examples_path);
  if (!reader) {
    LOG(ERROR) << "Cannot open examples file: " << examples_path;
    return 1;
  }
  auto writer = TFRecordWriter::New(outfile_path);
  if (!writer) {
    LOG(ERROR) << "Cannot open output file: " << outfile_path;
    return 1;
  }

  // ── P1: async writer thread ──────────────────────────────────────────────
  // Move CVO TFRecord writes off the main thread so we can overlap them
  // with the next batch's GPU compute. Bounded SPSC queue gives back-
  // pressure when writer falls behind the producer (rare since GPU is
  // much slower than disk write at our throughput).
  //
  // Design:
  //   main thread: build CVO → SerializeToString → enqueue
  //   writer thread: dequeue → writer->WriteRecord → loop
  //   end: main pushes 'done' flag, writer drains queue + exits
  //
  // Output bit-equivalence: writer thread is the SOLE consumer of the
  // writer; serialization order is preserved by the queue's FIFO
  // discipline. Same TFRecord bytes produced.
  constexpr size_t kWriteQueueDepth = 32;  // up to 32 CVOs buffered
  std::deque<std::string> write_queue;
  std::mutex wq_mu;
  std::condition_variable wq_nonempty, wq_nonfull;
  bool writer_done = false;
  std::atomic<bool> writer_failed{false};

  std::thread writer_thread([&]() {
    for (;;) {
      std::string item;
      {
        std::unique_lock<std::mutex> lk(wq_mu);
        wq_nonempty.wait(lk, [&] {
          return !write_queue.empty() || writer_done;
        });
        if (write_queue.empty() && writer_done) return;
        item = std::move(write_queue.front());
        write_queue.pop_front();
        wq_nonfull.notify_one();
      }
      if (!writer->WriteRecord(item)) {
        LOG(ERROR) << "Async writer: WriteRecord failed";
        writer_failed.store(true);
        // Drain remaining queue silently to unblock producer.
        std::lock_guard<std::mutex> lk(wq_mu);
        write_queue.clear();
        wq_nonfull.notify_all();
        return;
      }
    }
  });

  auto enqueue_write = [&](std::string&& payload) -> bool {
    if (writer_failed.load()) return false;
    std::unique_lock<std::mutex> lk(wq_mu);
    wq_nonfull.wait(lk, [&] {
      return write_queue.size() < kWriteQueueDepth || writer_failed.load();
    });
    if (writer_failed.load()) return false;
    write_queue.push_back(std::move(payload));
    wq_nonempty.notify_one();
    return true;
  };

  // Batch inference loop.
  int64_t total_examples = 0;
  int64_t total_batches  = 0;

  struct PendingExample {
    ExampleFeatures features;
    std::string raw_payload;  // original Example bytes (for passthrough fields)
  };
  std::vector<PendingExample> batch;
  batch.reserve(batch_size);

  // Hoist large per-batch buffer allocations out of the flush loop.
  // For batch_size=2048 and chr20 (B × H × W × C × 4 ≈ 1.3 GB), the
  // per-batch malloc + memset is a measurable cost (~80-150 ms per
  // batch on M4 Max). Allocate once at full capacity, reuse across
  // batches. The MPSGraph input wrapper reads only `n × elem` bytes
  // so the trailing slack is harmless.
  std::vector<float> images(static_cast<size_t>(batch_size) *
                              static_cast<size_t>(H * W * C));
  std::vector<float> probs(static_cast<size_t>(batch_size) *
                             static_cast<size_t>(K));

  auto flush_batch = [&]() -> bool {
    if (batch.empty()) return true;
    const int n = static_cast<int>(batch.size());
    const int64_t elem = H * W * C;
    DV_SIGNPOST_INTERVAL_BEGIN(FlushBatch, "");
    DV_SIGNPOST_INTERVAL_BEGIN(Normalize, "");
    for (int i = 0; i < n; ++i) {
      const std::string& img = batch[i].features.image_encoded;
      if (static_cast<int64_t>(img.size()) != elem) {
        // Try float32 layout (some variants store floats directly).
        if (static_cast<int64_t>(img.size()) == elem * 4) {
          std::memcpy(images.data() + i * elem, img.data(), elem * 4);
        } else {
          LOG(ERROR) << "Unexpected image size " << img.size()
                     << " (expected " << elem << " or " << elem * 4 << ")";
          return false;
        }
      } else {
        // uint8 → float32 normalized to [-1, 1] via (x - 128) / 128.
        // This matches the upstream DeepVariant preprocess_images (see
        // deepvariant/dv_utils.py: tf.subtract(images, 128.0); divide(., 128.0)).
        //
        // Bit-equivalence note: 1/128 = 2^-7 is exactly representable in
        // FP32, and (byte - 128.0f) for byte ∈ [0,255] is also exact, so
        // the multiplication produces exact results matching the scalar
        // path bit-for-bit. NEON intrinsics use IEEE 754 single-rounded
        // ops on Apple Silicon → identical FP32 outputs vs the scalar
        // loop. Verified: same inputs through scalar vs NEON paths
        // produce byte-identical `images` buffer.
        const uint8_t* src = reinterpret_cast<const uint8_t*>(img.data());
        float* dst = images.data() + i * elem;
        constexpr float kInvScale = 1.0f / 128.0f;
#if DV_HAVE_NEON
        const float32x4_t k128 = vdupq_n_f32(128.0f);
        const float32x4_t kinv = vdupq_n_f32(kInvScale);
        const int64_t simd_end = elem & ~int64_t{15};
        for (int64_t j = 0; j < simd_end; j += 16) {
          uint8x16_t b = vld1q_u8(src + j);
          // 16 u8 → 4×4 u32 → 4×4 f32 lanes.
          uint16x8_t lo16 = vmovl_u8(vget_low_u8(b));
          uint16x8_t hi16 = vmovl_u8(vget_high_u8(b));
          float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
          float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
          float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
          float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));
          vst1q_f32(dst + j +  0, vmulq_f32(vsubq_f32(f0, k128), kinv));
          vst1q_f32(dst + j +  4, vmulq_f32(vsubq_f32(f1, k128), kinv));
          vst1q_f32(dst + j +  8, vmulq_f32(vsubq_f32(f2, k128), kinv));
          vst1q_f32(dst + j + 12, vmulq_f32(vsubq_f32(f3, k128), kinv));
        }
        // Tail (< 16 trailing bytes).
        for (int64_t j = simd_end; j < elem; ++j) {
          dst[j] = (static_cast<float>(src[j]) - 128.0f) * kInvScale;
        }
#else
        for (int64_t j = 0; j < elem; ++j) {
          dst[j] = (static_cast<float>(src[j]) - 128.0f) * kInvScale;
        }
#endif
      }
    }

    DV_SIGNPOST_INTERVAL_END(Normalize);

    // Run inference. (probs hoisted, see top of fn; features lazily
    // allocated to full batch capacity inside the metal branch.)
    bool ok = false;
    DV_SIGNPOST_INTERVAL_BEGIN(Inference, "");
    const bool ane_speculate_mode =
        (coreml_model && metal_model && metal_finalize);
    if (ane_speculate_mode) {
      // Scenario 3: ANE FP16 forward on the full batch; rerun
      // borderline-confidence examples on GPU MPSGraph FP32 +
      // BNNS-CPU finalize so threshold sites stay on the
      // deterministic FP32 path.
      DV_SIGNPOST_INTERVAL_BEGIN(AneFp16, "");
      ok = coreml_model->Predict(images.data(), n, H, W, C,
                                  probs.data(), K);
      DV_SIGNPOST_INTERVAL_END(AneFp16);
      if (ok) {
        // Identify borderline examples (max softmax < threshold).
        const float conf_threshold = static_cast<float>(
            absl::GetFlag(FLAGS_ane_speculate_confidence));
        static thread_local std::vector<int> borderline_idx;
        borderline_idx.clear();
        borderline_idx.reserve(n);
        for (int i = 0; i < n; ++i) {
          float m = probs[i * K];
          for (int j = 1; j < K; ++j) {
            if (probs[i * K + j] > m) m = probs[i * K + j];
          }
          if (m < conf_threshold) borderline_idx.push_back(i);
        }
        if (!borderline_idx.empty()) {
          DV_SIGNPOST_INTERVAL_BEGIN(AneRerunGpu, "");
          const int nb = static_cast<int>(borderline_idx.size());
          const size_t img_per = static_cast<size_t>(H) * W * C;
          static thread_local std::vector<float> bl_images, bl_features,
              bl_probs;
          bl_images.resize(static_cast<size_t>(nb) * img_per);
          bl_features.resize(static_cast<size_t>(nb) *
                             metal_model->FeatureDim());
          bl_probs.resize(static_cast<size_t>(nb) * K);
          for (int b = 0; b < nb; ++b) {
            const int src = borderline_idx[b];
            std::memcpy(bl_images.data() + static_cast<size_t>(b) * img_per,
                        images.data() + static_cast<size_t>(src) * img_per,
                        img_per * sizeof(float));
          }
          bool gpu_ok = metal_model->Predict(bl_images.data(), nb,
                                              bl_features.data());
          if (gpu_ok) {
            gpu_ok = metal_finalize->ApplyBatch(bl_features.data(), nb,
                                                bl_probs.data());
          }
          if (gpu_ok) {
            for (int b = 0; b < nb; ++b) {
              const int dst = borderline_idx[b];
              std::memcpy(probs.data() + static_cast<size_t>(dst) * K,
                          bl_probs.data() + static_cast<size_t>(b) * K,
                          K * sizeof(float));
            }
          } else {
            ok = false;
            LOG(ERROR) << "ane_speculate: GPU rerun failed on "
                       << nb << " borderline examples";
          }
          DV_SIGNPOST_INTERVAL_END(AneRerunGpu);
        }
      }
    } else if (coreml_model) {
      ok = coreml_model->Predict(images.data(), n, H, W, C,
                                  probs.data(), K);
    } else if (metal_model && metal_model->IsGpuFinalize()) {
      // Single-stage GPU path (DV_METAL_GPU_FINALIZE=1): the dense +
      // softmax run inside MPSGraph, so Predict() writes (n, 3)
      // probabilities directly. metal_finalize is unused in this mode.
      DV_SIGNPOST_INTERVAL_BEGIN(MetalGPU, "");
      ok = metal_model->Predict(images.data(), n, probs.data());
      DV_SIGNPOST_INTERVAL_END(MetalGPU);
    } else if (metal_model && metal_finalize) {
      // Two-stage Metal/BNNS path: GPU MPSGraph for backbone, CPU BNNS
      // for the final dense + softmax (deterministic FP32 reduction
      // = bit-parity with TF CPU). features sized to full batch_size
      // on first use; subsequent batches reuse via static thread-local.
      static thread_local std::vector<float> features;
      const size_t feat_total = static_cast<size_t>(batch_size) *
                                  static_cast<size_t>(metal_model->FeatureDim());
      if (features.size() < feat_total) features.resize(feat_total);
      DV_SIGNPOST_INTERVAL_BEGIN(MetalGPU, "");
      bool gpu_ok = metal_model->Predict(images.data(), n, features.data());
      DV_SIGNPOST_INTERVAL_END(MetalGPU);
      if (gpu_ok) {
        DV_SIGNPOST_INTERVAL_BEGIN(BnnsFinalize, "");
        ok = metal_finalize->ApplyBatch(features.data(), n, probs.data());
        DV_SIGNPOST_INTERVAL_END(BnnsFinalize);
      }
    }
    DV_SIGNPOST_INTERVAL_END(Inference);
    if (!ok) {
      LOG(ERROR) << "Inference failed on batch " << total_batches;
      return false;
    }

    // Write one CallVariantsOutput per example.
    for (int i = 0; i < n; ++i) {
      learning::genomics::deepvariant::CallVariantsOutput cvo;
      if (!batch[i].features.variant_encoded.empty()) {
        cvo.mutable_variant()->ParseFromString(
            batch[i].features.variant_encoded);
      }
      if (!batch[i].features.alt_allele_indices_encoded.empty()) {
        cvo.mutable_alt_allele_indices()->ParseFromString(
            batch[i].features.alt_allele_indices_encoded);
      }
      for (int k = 0; k < K; ++k) {
        cvo.add_genotype_probabilities(probs[i * K + k]);
      }
      // Tag MID="deepvariant" so postprocess can write it as a VCF FORMAT
      // field. Reuse the empty VariantCall slot that variant_calling.cc
      // already added (otherwise we end up with 2 calls and VcfWriter
      // rejects the variant for not matching sample count).
      auto* v = cvo.mutable_variant();
      if (v->calls_size() == 0) v->add_calls();
      nucleus::SetInfoField("MID", std::string("deepvariant"),
                             v->mutable_calls(0));

      std::string serialized;
      if (!cvo.SerializeToString(&serialized)) {
        LOG(ERROR) << "Failed to serialize CallVariantsOutput";
        return false;
      }
      // P1: async writer thread consumes this. Push std::move so the
      // writer thread owns the buffer; main thread can recycle storage.
      if (!enqueue_write(std::move(serialized))) {
        LOG(ERROR) << "Failed to enqueue output record (writer thread error)";
        return false;
      }
    }

    ++total_batches;
    total_examples += n;
    batch.clear();
    DV_SIGNPOST_INTERVAL_END(FlushBatch);
    return true;
  };

  // ── P2: pre-fetch reader thread ──────────────────────────────────────────
  // Move reader->GetNext() + ParseExample off the main thread so we can
  // overlap the I/O + protobuf parsing with the previous batch's GPU
  // dispatch. Bounded SPSC queue (depth = 2 × batch_size = 1024 examples
  // at default batch=512) gives back-pressure when main thread is the
  // bottleneck.
  //
  // Output bit-equivalence: reader produces same PendingExample objects
  // in the same order; main thread consumes in same order; flush_batch
  // sees identical batches as before. No algorithmic change.
  const size_t kReadQueueDepth = static_cast<size_t>(batch_size) * 2;
  std::deque<PendingExample> read_queue;
  std::mutex rq_mu;
  std::condition_variable rq_nonempty, rq_nonfull;
  bool reader_eof = false;
  std::atomic<bool> reader_stop{false};

  std::thread reader_thread([&]() {
    while (!reader_stop.load() && reader->GetNext()) {
      PendingExample pe;
      pe.raw_payload = reader->record();
      pe.features    = ParseExample(pe.raw_payload);
      std::unique_lock<std::mutex> lk(rq_mu);
      rq_nonfull.wait(lk, [&] {
        return read_queue.size() < kReadQueueDepth || reader_stop.load();
      });
      if (reader_stop.load()) return;
      read_queue.push_back(std::move(pe));
      rq_nonempty.notify_one();
    }
    {
      std::lock_guard<std::mutex> lk(rq_mu);
      reader_eof = true;
    }
    rq_nonempty.notify_all();
  });

  // RAII guard: ensure reader thread is joined on every exit path.
  struct ReaderJoiner {
    std::thread& t;
    std::atomic<bool>& stop;
    std::mutex& mu;
    std::condition_variable& cv_full;
    std::condition_variable& cv_empty;
    ~ReaderJoiner() {
      stop.store(true);
      { std::lock_guard<std::mutex> lk(mu); }
      cv_full.notify_all();
      cv_empty.notify_all();
      if (t.joinable()) t.join();
    }
  } reader_joiner{reader_thread, reader_stop, rq_mu, rq_nonfull, rq_nonempty};

  // Main consumption loop: pop from reader queue, accumulate batch,
  // flush when full.
  for (;;) {
    PendingExample pe;
    bool got_one = false;
    {
      std::unique_lock<std::mutex> lk(rq_mu);
      rq_nonempty.wait(lk, [&] {
        return !read_queue.empty() || reader_eof;
      });
      if (!read_queue.empty()) {
        pe = std::move(read_queue.front());
        read_queue.pop_front();
        rq_nonfull.notify_one();
        got_one = true;
      } else if (reader_eof) {
        break;
      }
    }
    if (got_one) {
      batch.push_back(std::move(pe));
      if (static_cast<int>(batch.size()) >= batch_size) {
        if (!flush_batch()) return 1;
      }
    }
  }
  if (!flush_batch()) return 1;

  // Signal writer thread to drain + exit; then close writer ourselves.
  {
    std::lock_guard<std::mutex> lk(wq_mu);
    writer_done = true;
  }
  wq_nonempty.notify_all();
  writer_thread.join();
  if (writer_failed.load()) {
    LOG(ERROR) << "Async writer thread failed during run";
    return 1;
  }

  reader->Close();
  writer->Close();

  LOG(INFO) << "call_variants done: " << total_examples << " examples, "
            << total_batches << " batches → " << outfile_path;
  return 0;
}

}  // namespace deepvariant
