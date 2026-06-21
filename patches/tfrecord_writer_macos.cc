// POSIX replacement for third_party/nucleus/io/tfrecord_writer.cc.
// Format: [uint64_le length][uint32_le masked_crc32c(len)]
//         [bytes payload][uint32_le masked_crc32c(payload)]

#include "third_party/nucleus/io/tfrecord_writer.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "absl/crc/crc32c.h"

namespace nucleus {

namespace {
constexpr uint32_t kMaskDelta = 0xa282ead8UL;
uint32_t MaskedCrc32c(const char* data, size_t n) {
  uint32_t crc = static_cast<uint32_t>(
      absl::ComputeCrc32c(std::string_view(data, n)));
  return ((crc >> 15) | (crc << 17)) + kMaskDelta;
}

struct TFRWImpl { std::ofstream stream; };
std::mutex mu;
std::unordered_map<TFRecordWriter*, std::unique_ptr<TFRWImpl>> impls;
}  // namespace

TFRecordWriter::TFRecordWriter()  = default;
TFRecordWriter::~TFRecordWriter() {
  std::lock_guard<std::mutex> lk(mu);
  impls.erase(this);
}

// static
std::unique_ptr<TFRecordWriter> TFRecordWriter::New(
    const std::string& filename, const std::string& /*compression_type*/) {
  auto impl = std::make_unique<TFRWImpl>();
  impl->stream.open(filename, std::ios::binary | std::ios::trunc);
  if (!impl->stream.is_open()) return nullptr;
  auto w = std::unique_ptr<TFRecordWriter>(new TFRecordWriter());
  {
    std::lock_guard<std::mutex> lk(mu);
    impls[w.get()] = std::move(impl);
  }
  return w;
}

bool TFRecordWriter::WriteRecord(const std::string& record) {
  std::lock_guard<std::mutex> lk(mu);
  auto it = impls.find(this);
  if (it == impls.end()) return false;
  auto& s = it->second->stream;

  uint64_t len = record.size();
  uint32_t len_crc = MaskedCrc32c(reinterpret_cast<const char*>(&len), 8);
  uint32_t data_crc = MaskedCrc32c(record.data(), len);

  s.write(reinterpret_cast<const char*>(&len), 8);
  s.write(reinterpret_cast<const char*>(&len_crc), 4);
  s.write(record.data(), static_cast<std::streamsize>(len));
  s.write(reinterpret_cast<const char*>(&data_crc), 4);
  return s.good();
}

bool TFRecordWriter::Flush() {
  std::lock_guard<std::mutex> lk(mu);
  auto it = impls.find(this);
  if (it == impls.end()) return false;
  it->second->stream.flush();
  return it->second->stream.good();
}

bool TFRecordWriter::Close() {
  std::lock_guard<std::mutex> lk(mu);
  auto it = impls.find(this);
  if (it == impls.end()) return true;
  it->second->stream.flush();
  it->second->stream.close();
  return !it->second->stream.fail();
}

}  // namespace nucleus
