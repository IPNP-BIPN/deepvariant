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

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "deepvariant/native/coreml_inference.h"
#include "deepvariant/native/tfrecord.h"
#include "deepvariant/protos/deepvariant.pb.h"
#include "third_party/nucleus/protos/variants.pb.h"

ABSL_FLAG(std::string, examples,  "", "Input TFRecord file(s) of tf.train.Example.");
ABSL_FLAG(std::string, checkpoint, "", ".mlpackage path for the inference model.");
ABSL_FLAG(std::string, outfile,   "", "Output TFRecord file for CallVariantsOutput.");
ABSL_FLAG(int,    batch_size, 128, "Inference batch size.");
ABSL_FLAG(std::string, compute_units, "all",
          "Core ML compute units: all (default), cpu_gpu, cpu_only.");

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

  // Load model.
  LOG(INFO) << "Loading model: " << checkpoint_path;
  auto model = CoreMLModel::Load(checkpoint_path, compute_units);
  if (!model) {
    LOG(ERROR) << "Failed to load Core ML model: " << checkpoint_path;
    return 1;
  }
  const int H = model->InputHeight();
  const int W = model->InputWidth();
  const int C = model->InputChannels();
  const int K = model->NumClasses();
  LOG(INFO) << "Model input (" << H << "," << W << "," << C
            << ") → " << K << " classes";

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

  // Batch inference loop.
  int64_t total_examples = 0;
  int64_t total_batches  = 0;

  struct PendingExample {
    ExampleFeatures features;
    std::string raw_payload;  // original Example bytes (for passthrough fields)
  };
  std::vector<PendingExample> batch;
  batch.reserve(batch_size);

  auto flush_batch = [&]() -> bool {
    if (batch.empty()) return true;
    const int n = static_cast<int>(batch.size());
    const int64_t elem = H * W * C;

    // Pack images into flat float32 buffer.
    std::vector<float> images(n * elem);
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
        const uint8_t* src = reinterpret_cast<const uint8_t*>(img.data());
        float* dst = images.data() + i * elem;
        constexpr float kInvScale = 1.0f / 128.0f;
        for (int64_t j = 0; j < elem; ++j) {
          dst[j] = (static_cast<float>(src[j]) - 128.0f) * kInvScale;
        }
      }
    }

    // Run inference.
    std::vector<float> probs(n * K);
    if (!model->Predict(images.data(), n, H, W, C, probs.data(), K)) {
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
      std::string serialized;
      if (!cvo.SerializeToString(&serialized)) {
        LOG(ERROR) << "Failed to serialize CallVariantsOutput";
        return false;
      }
      if (!writer->WriteRecord(serialized)) {
        LOG(ERROR) << "Failed to write output record";
        return false;
      }
    }

    ++total_batches;
    total_examples += n;
    batch.clear();
    return true;
  };

  while (reader->GetNext()) {
    PendingExample pe;
    pe.raw_payload = reader->record();
    pe.features    = ParseExample(pe.raw_payload);
    batch.push_back(std::move(pe));

    if (static_cast<int>(batch.size()) >= batch_size) {
      if (!flush_batch()) return 1;
    }
  }
  if (!flush_batch()) return 1;

  reader->Close();
  writer->Close();

  LOG(INFO) << "call_variants done: " << total_examples << " examples, "
            << total_batches << " batches → " << outfile_path;
  return 0;
}

}  // namespace deepvariant
