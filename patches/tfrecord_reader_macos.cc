// POSIX replacement for third_party/nucleus/io/tfrecord_reader.cc.
// TFRecord format: [uint64_le length][uint32_le masked_crc32c(len)]
//                  [bytes payload][uint32_le masked_crc32c(payload)]
// Uncompressed only. CRC not verified (dev-speed path).

#include "third_party/nucleus/io/tfrecord_reader.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace nucleus {

namespace {
struct TFRRImpl {
  std::ifstream stream;
  explicit TFRRImpl(const std::string& path)
      : stream(path, std::ios::binary) {}
};
std::mutex mu;
std::unordered_map<TFRecordReader*, std::unique_ptr<TFRRImpl>> impls;
}  // namespace

TFRecordReader::TFRecordReader() : offset_(0) {}
TFRecordReader::~TFRecordReader() {
  std::lock_guard<std::mutex> lk(mu);
  impls.erase(this);
}

// static
std::unique_ptr<TFRecordReader> TFRecordReader::New(
    const std::string& filename, const std::string& /*compression_type*/) {
  auto impl = std::make_unique<TFRRImpl>(filename);
  if (!impl->stream.is_open()) return nullptr;
  auto r = std::unique_ptr<TFRecordReader>(new TFRecordReader());
  {
    std::lock_guard<std::mutex> lk(mu);
    impls[r.get()] = std::move(impl);
  }
  return r;
}

bool TFRecordReader::GetNext() {
  std::lock_guard<std::mutex> lk(mu);
  auto it = impls.find(this);
  if (it == impls.end()) return false;
  auto& s = it->second->stream;
  if (!s.good()) return false;

  uint64_t length = 0;
  s.read(reinterpret_cast<char*>(&length), 8);
  if (s.gcount() != 8) return false;
  s.seekg(4, std::ios::cur);  // skip length CRC

  record_.resize(length);
  s.read(record_.data(), static_cast<std::streamsize>(length));
  if (static_cast<uint64_t>(s.gcount()) != length) return false;
  s.seekg(4, std::ios::cur);  // skip payload CRC

  offset_ += 8 + 4 + length + 4;
  return true;
}

void TFRecordReader::Close() {
  std::lock_guard<std::mutex> lk(mu);
  auto it = impls.find(this);
  if (it != impls.end()) it->second->stream.close();
}

}  // namespace nucleus
