// TFRecord reader/writer for deepvariant native runtime.
// See tfrecord.h for the format description.

#include "deepvariant/native/tfrecord.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/crc/crc32c.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

namespace deepvariant {

namespace {
constexpr uint32_t kMaskDelta = 0xa282ead8UL;

uint32_t MaskedCrc32c(const char* data, size_t n) {
  uint32_t crc = static_cast<uint32_t>(absl::ComputeCrc32c({data, n}));
  return ((crc >> 15) | (crc << 17)) + kMaskDelta;
}

// Expand `prefix@N` to {prefix-00000-of-NNNNN, ..., prefix-(N-1)-of-NNNNN}.
// Plain paths (no `@`) pass through as a single-element list.
std::vector<std::string> ExpandShards(const std::string& spec) {
  auto at = spec.find('@');
  if (at == std::string::npos) return {spec};
  const std::string prefix = spec.substr(0, at);
  int n = 0;
  if (!absl::SimpleAtoi(spec.substr(at + 1), &n) || n <= 0) return {spec};
  std::vector<std::string> paths;
  paths.reserve(n);
  for (int i = 0; i < n; ++i) {
    paths.push_back(absl::StrCat(prefix, "-",
                                  absl::Dec(i, absl::kZeroPad5),
                                  "-of-", absl::Dec(n, absl::kZeroPad5)));
  }
  return paths;
}
}  // namespace

std::string ShardName(const std::string& spec, int task_id) {
  auto at = spec.find('@');
  if (at == std::string::npos) return spec;
  const std::string prefix = spec.substr(0, at);
  int n = 0;
  if (!absl::SimpleAtoi(spec.substr(at + 1), &n) || n <= 0) return spec;
  return absl::StrCat(prefix, "-", absl::Dec(task_id, absl::kZeroPad5),
                       "-of-", absl::Dec(n, absl::kZeroPad5));
}

// ---------------------------------------------------------------------------
// TFRecordReader
// ---------------------------------------------------------------------------

struct TFRecordReader::Impl {
  std::vector<std::string> paths;
  size_t current_index = 0;
  std::ifstream stream;

  explicit Impl(const std::string& spec) : paths(ExpandShards(spec)) {
    if (!paths.empty()) stream.open(paths[0], std::ios::binary);
  }

  // Advance to the next shard if the current one is exhausted; returns true
  // if a stream is currently open and ready for reading.
  bool EnsureOpen() {
    if (stream.is_open() && stream.good()) return true;
    while (current_index + 1 < paths.size()) {
      stream.close();
      ++current_index;
      stream.clear();
      stream.open(paths[current_index], std::ios::binary);
      if (stream.is_open() && stream.good()) return true;
    }
    return false;
  }
};

TFRecordReader::TFRecordReader() = default;
TFRecordReader::~TFRecordReader() = default;

std::unique_ptr<TFRecordReader> TFRecordReader::New(
    const std::string& path, const std::string& /*compression_type*/) {
  auto impl = std::make_unique<Impl>(path);
  if (impl->paths.empty()) return nullptr;
  if (!impl->stream.is_open()) return nullptr;
  auto r = std::unique_ptr<TFRecordReader>(new TFRecordReader());
  r->impl_ = std::move(impl);
  return r;
}

bool TFRecordReader::GetNext() {
  if (!impl_) return false;
  while (true) {
    auto& s = impl_->stream;
    if (s.good()) {
      uint64_t length = 0;
      s.read(reinterpret_cast<char*>(&length), 8);
      if (s.gcount() == 8) {
        s.seekg(4, std::ios::cur);  // skip length CRC (not verified)

        record_.resize(length);
        s.read(record_.data(), static_cast<std::streamsize>(length));
        if (static_cast<uint64_t>(s.gcount()) != length) return false;
        s.seekg(4, std::ios::cur);  // skip payload CRC

        offset_ += 8 + 4 + length + 4;
        return true;
      }
    }
    // Current shard exhausted (or read failed at boundary). Try next shard.
    if (impl_->current_index + 1 >= impl_->paths.size()) return false;
    impl_->stream.close();
    ++impl_->current_index;
    impl_->stream.clear();
    impl_->stream.open(impl_->paths[impl_->current_index], std::ios::binary);
    if (!impl_->stream.is_open()) return false;
    offset_ = 0;
  }
}

void TFRecordReader::Close() {
  if (impl_) impl_->stream.close();
}

// ---------------------------------------------------------------------------
// TFRecordWriter
// ---------------------------------------------------------------------------

struct TFRecordWriter::Impl {
  std::ofstream stream;
  explicit Impl(const std::string& path)
      : stream(path, std::ios::binary | std::ios::trunc) {}
};

TFRecordWriter::TFRecordWriter() = default;
TFRecordWriter::~TFRecordWriter() = default;

std::unique_ptr<TFRecordWriter> TFRecordWriter::New(
    const std::string& path, const std::string& /*compression_type*/) {
  auto impl = std::make_unique<Impl>(path);
  if (!impl->stream.is_open()) return nullptr;
  auto w = std::unique_ptr<TFRecordWriter>(new TFRecordWriter());
  w->impl_ = std::move(impl);
  return w;
}

bool TFRecordWriter::WriteRecord(const std::string& payload) {
  if (!impl_ || !impl_->stream.good()) return false;
  auto& s = impl_->stream;

  uint64_t len = payload.size();
  uint32_t len_crc =
      MaskedCrc32c(reinterpret_cast<const char*>(&len), sizeof(len));
  uint32_t data_crc = MaskedCrc32c(payload.data(), len);

  s.write(reinterpret_cast<const char*>(&len), 8);
  s.write(reinterpret_cast<const char*>(&len_crc), 4);
  s.write(payload.data(), static_cast<std::streamsize>(len));
  s.write(reinterpret_cast<const char*>(&data_crc), 4);
  return s.good();
}

bool TFRecordWriter::Flush() {
  if (!impl_) return false;
  impl_->stream.flush();
  return impl_->stream.good();
}

bool TFRecordWriter::Close() {
  if (!impl_) return true;
  impl_->stream.flush();
  impl_->stream.close();
  return !impl_->stream.fail();
}

}  // namespace deepvariant
