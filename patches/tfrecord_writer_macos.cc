// POSIX + protobuf replacement for third_party/nucleus/io/tfrecord_writer.cc.
// Writes TFRecords without TF runtime. CRC uses the crc32c implementation from
// abseil (absl::crc::crc32c::Extend).
// Format: [uint64_le length][uint32_le masked_crc32c(len)]
//         [bytes payload][uint32_le masked_crc32c(payload)]

#include "third_party/nucleus/io/tfrecord_writer.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

#include "absl/crc/crc32c.h"

namespace nucleus {

namespace {
// masked CRC32C — matches TF's RecordWriter encoding.
constexpr uint32_t kMaskDelta = 0xa282ead8UL;
uint32_t MaskedCrc32c(const char* data, size_t n) {
  uint32_t crc = static_cast<uint32_t>(
      absl::ComputeCrc32c(std::string_view(data, n)));
  return ((crc >> 15) | (crc << 17)) + kMaskDelta;
}
}  // namespace

struct TFRecordWriterImpl {
  std::ofstream stream;
  explicit TFRecordWriterImpl(const std::string& path)
      : stream(path, std::ios::binary | std::ios::trunc) {}
};

TFRecordWriter::TFRecordWriter() = default;
TFRecordWriter::~TFRecordWriter() = default;

// static
std::unique_ptr<TFRecordWriter> TFRecordWriter::New(
    const std::string& filename, const std::string& /*compression_type*/) {
  auto w = std::unique_ptr<TFRecordWriter>(new TFRecordWriter());
  auto impl = std::make_unique<TFRecordWriterImpl>(filename);
  if (!impl->stream.is_open()) return nullptr;
  w->file_ = std::move(impl);
  return w;
}

bool TFRecordWriter::WriteRecord(const std::string& record) {
  auto* impl = static_cast<TFRecordWriterImpl*>(file_.get());
  if (!impl || !impl->stream.good()) return false;

  uint64_t len = record.size();
  uint32_t len_crc = MaskedCrc32c(reinterpret_cast<const char*>(&len), 8);
  uint32_t data_crc = MaskedCrc32c(record.data(), len);

  impl->stream.write(reinterpret_cast<const char*>(&len), 8);
  impl->stream.write(reinterpret_cast<const char*>(&len_crc), 4);
  impl->stream.write(record.data(), static_cast<std::streamsize>(len));
  impl->stream.write(reinterpret_cast<const char*>(&data_crc), 4);
  return impl->stream.good();
}

bool TFRecordWriter::Flush() {
  if (!file_) return false;
  auto* impl = static_cast<TFRecordWriterImpl*>(file_.get());
  impl->stream.flush();
  return impl->stream.good();
}

bool TFRecordWriter::Close() {
  if (!file_) return true;
  auto* impl = static_cast<TFRecordWriterImpl*>(file_.get());
  impl->stream.flush();
  impl->stream.close();
  return !impl->stream.fail();
}

}  // namespace nucleus
