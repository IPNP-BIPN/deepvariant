// TFRecord reader/writer for deepvariant native runtime.
// See tfrecord.h for the format description.

#include "deepvariant/native/tfrecord.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>

#include "absl/crc/crc32c.h"

namespace deepvariant {

namespace {
constexpr uint32_t kMaskDelta = 0xa282ead8UL;

uint32_t MaskedCrc32c(const char* data, size_t n) {
  uint32_t crc = static_cast<uint32_t>(absl::ComputeCrc32c({data, n}));
  return ((crc >> 15) | (crc << 17)) + kMaskDelta;
}
}  // namespace

// ---------------------------------------------------------------------------
// TFRecordReader
// ---------------------------------------------------------------------------

struct TFRecordReader::Impl {
  std::ifstream stream;
  explicit Impl(const std::string& path)
      : stream(path, std::ios::binary) {}
};

TFRecordReader::TFRecordReader() = default;
TFRecordReader::~TFRecordReader() = default;

std::unique_ptr<TFRecordReader> TFRecordReader::New(
    const std::string& path, const std::string& /*compression_type*/) {
  auto impl = std::make_unique<Impl>(path);
  if (!impl->stream.is_open()) return nullptr;
  auto r = std::unique_ptr<TFRecordReader>(new TFRecordReader());
  r->impl_ = std::move(impl);
  return r;
}

bool TFRecordReader::GetNext() {
  if (!impl_ || !impl_->stream.good()) return false;
  auto& s = impl_->stream;

  uint64_t length = 0;
  s.read(reinterpret_cast<char*>(&length), 8);
  if (s.gcount() != 8) return false;
  s.seekg(4, std::ios::cur);  // skip length CRC (not verified)

  record_.resize(length);
  s.read(record_.data(), static_cast<std::streamsize>(length));
  if (static_cast<uint64_t>(s.gcount()) != length) return false;
  s.seekg(4, std::ios::cur);  // skip payload CRC

  offset_ += 8 + 4 + length + 4;
  return true;
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
