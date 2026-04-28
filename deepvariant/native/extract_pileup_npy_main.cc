// Profiling tool: extract the first N pileup images from a TFRecord (or
// `name@N` shard spec) and write them as a NumPy `.npy` array of shape
// (N, 100, 221, 7) FP32 NHWC.  Pixel encoding mirrors call_variants:
//   uint8 src → (src - 128) / 128.0 → FP32
// or a passthrough when the input is already FP32.
//
// Used by Phase 5.5c per-layer drift profiling: produces a real-data
// `_input.npy` that `dump_tf_per_layer.py` (Docker) and
// `debug_metal --compare-to-reference` both consume.
//
// usage:
//   extract_pileup_npy <examples.tfrecord[@N]> <out.npy> [count=64]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "deepvariant/native/tfrecord.h"

namespace {

// Minimal protobuf wire decoders — ported from call_variants_main.cc's
// anonymous namespace (we don't link the runtime here).

uint64_t ReadVarint(const uint8_t* buf, size_t len, size_t& i) {
  uint64_t val = 0;
  int shift = 0;
  while (i < len) {
    uint8_t b = buf[i++];
    val |= static_cast<uint64_t>(b & 0x7F) << shift;
    if (!(b & 0x80)) return val;
    shift += 7;
  }
  return val;
}

std::string ExtractBytesListFirst(const uint8_t* buf, size_t len) {
  size_t i = 0;
  while (i < len) {
    uint64_t tag = ReadVarint(buf, len, i);
    uint32_t field = static_cast<uint32_t>(tag >> 3);
    uint32_t wire  = static_cast<uint32_t>(tag & 7);
    if (wire != 2) break;
    uint64_t seg_len = ReadVarint(buf, len, i);
    if (i + seg_len > len) break;
    if (field == 1) {
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

std::string ParseImageEncoded(const std::string& payload) {
  const uint8_t* buf = reinterpret_cast<const uint8_t*>(payload.data());
  size_t n = payload.size();
  size_t i = 0;
  while (i < n) {
    uint64_t tag = ReadVarint(buf, n, i);
    uint32_t wire = tag & 7;
    if (wire != 2) break;
    uint64_t seg_len = ReadVarint(buf, n, i);
    if (i + seg_len > n) break;
    const uint8_t* feat_buf = buf + i;
    size_t feat_len = seg_len;
    i += seg_len;
    size_t fi = 0;
    while (fi < feat_len) {
      uint64_t ftag = ReadVarint(feat_buf, feat_len, fi);
      if ((ftag & 7) != 2) break;
      uint64_t entry_len = ReadVarint(feat_buf, feat_len, fi);
      if (fi + entry_len > feat_len) break;
      const uint8_t* entry = feat_buf + fi;
      fi += entry_len;
      std::string key;
      std::string value_bytes;
      size_t ei = 0;
      while (ei < entry_len) {
        uint64_t etag = ReadVarint(entry, entry_len, ei);
        uint32_t efd = etag >> 3;
        if ((etag & 7) != 2) break;
        uint64_t elen = ReadVarint(entry, entry_len, ei);
        if (ei + elen > entry_len) break;
        if (efd == 1) {
          key.assign(reinterpret_cast<const char*>(entry + ei), elen);
        } else if (efd == 2) {
          value_bytes.assign(reinterpret_cast<const char*>(entry + ei), elen);
        }
        ei += elen;
      }
      if (key == "image/encoded" || key == "image") {
        return ExtractBytesListFirst(
            reinterpret_cast<const uint8_t*>(value_bytes.data()),
            value_bytes.size());
      }
    }
  }
  return {};
}

// Write a (N, 100, 221, 7) FP32 NHWC array to NumPy v1 .npy.
bool WriteNpyFp32NHWC(const std::string& path, int N, int H, int W, int C,
                      const float* data) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;

  std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
      std::to_string(N) + ", " + std::to_string(H) + ", " +
      std::to_string(W) + ", " + std::to_string(C) + "), }";
  // Pad header so total prefix (10 bytes magic+version+len + header + 1 \n)
  // is a multiple of 64 — required by the .npy format.
  while (((10 + header.size() + 1) % 64) != 0) header.push_back(' ');
  header.push_back('\n');

  const char magic[6] = {'\x93','N','U','M','P','Y'};
  f.write(magic, 6);
  uint8_t major = 1, minor = 0;
  f.write(reinterpret_cast<const char*>(&major), 1);
  f.write(reinterpret_cast<const char*>(&minor), 1);
  uint16_t hl = static_cast<uint16_t>(header.size());
  f.write(reinterpret_cast<const char*>(&hl), 2);
  f.write(header.data(), header.size());

  const size_t n_bytes =
      static_cast<size_t>(N) * H * W * C * sizeof(float);
  f.write(reinterpret_cast<const char*>(data), n_bytes);
  return f.good();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr,
        "usage: %s <examples.tfrecord[@N]> <out.npy> [count=64]\n", argv[0]);
    return 2;
  }
  const std::string tfr_path = argv[1];
  const std::string out_path = argv[2];
  const int count = (argc >= 4) ? std::atoi(argv[3]) : 64;
  if (count <= 0 || count > 100000) {
    std::fprintf(stderr, "bad count=%d\n", count);
    return 2;
  }

  // Standard WGS DeepVariant pileup geometry.
  constexpr int H = 100, W = 221, C = 7;
  constexpr int64_t kElemPerImg = static_cast<int64_t>(H) * W * C;

  auto reader = deepvariant::TFRecordReader::New(tfr_path);
  if (!reader) {
    std::fprintf(stderr, "cannot open %s\n", tfr_path.c_str());
    return 1;
  }

  std::vector<float> all(static_cast<size_t>(count) * kElemPerImg);
  int n_loaded = 0;
  for (int i = 0; i < count; ++i) {
    if (!reader->GetNext()) {
      std::fprintf(stderr, "EOF after %d records\n", i);
      break;
    }
    const std::string img = ParseImageEncoded(reader->record());
    float* dst = all.data() + static_cast<size_t>(i) * kElemPerImg;
    if (static_cast<int64_t>(img.size()) == kElemPerImg) {
      // uint8 → (x - 128) / 128 — same path as call_variants.
      const uint8_t* src = reinterpret_cast<const uint8_t*>(img.data());
      constexpr float inv = 1.0f / 128.0f;
      for (int64_t j = 0; j < kElemPerImg; ++j) {
        dst[j] = (static_cast<float>(src[j]) - 128.0f) * inv;
      }
    } else if (static_cast<int64_t>(img.size()) == kElemPerImg * 4) {
      std::memcpy(dst, img.data(),
                  static_cast<size_t>(kElemPerImg) * sizeof(float));
    } else {
      std::fprintf(stderr,
          "record %d: bad image size %zu (expected %lld or %lld)\n",
          i, img.size(),
          static_cast<long long>(kElemPerImg),
          static_cast<long long>(kElemPerImg * 4));
      return 1;
    }
    ++n_loaded;
  }

  if (!WriteNpyFp32NHWC(out_path, n_loaded, H, W, C, all.data())) {
    std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
    return 1;
  }
  std::printf("wrote %d images to %s (shape %d×%d×%d×%d, %.1f MB)\n",
              n_loaded, out_path.c_str(), n_loaded, H, W, C,
              n_loaded * kElemPerImg * 4.0 / (1024.0 * 1024.0));
  return 0;
}
