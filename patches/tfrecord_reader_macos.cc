// POSIX + protobuf replacement for third_party/nucleus/io/tfrecord_reader.cc.
// TFRecord format: [uint64_le length][uint32_le masked_crc32c(len)]
//                  [bytes payload][uint32_le masked_crc32c(payload)]
// Uncompressed only — DeepVariant makes_examples always writes uncompressed.
// CRC is not verified in this implementation (dev-speed path).

#include "third_party/nucleus/io/tfrecord_reader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

namespace nucleus {

struct TFRecordReaderImpl {
  std::ifstream stream;
  explicit TFRecordReaderImpl(const std::string& path) : stream(path, std::ios::binary) {}
};

TFRecordReader::TFRecordReader() : offset_(0) {}
TFRecordReader::~TFRecordReader() = default;

// static
std::unique_ptr<TFRecordReader> TFRecordReader::New(
    const std::string& filename, const std::string& /*compression_type*/) {
  auto r = std::unique_ptr<TFRecordReader>(new TFRecordReader());
  auto impl = std::make_unique<TFRecordReaderImpl>(filename);
  if (!impl->stream.is_open()) return nullptr;
  // Abuse reader_ pointer to store our impl (void* via reinterpret cast trick;
  // we store it in file_ using the same underlying storage pattern).
  r->file_ = std::move(impl);  // store as the "file" unique_ptr
  return r;
}

bool TFRecordReader::GetNext() {
  auto* impl = static_cast<TFRecordReaderImpl*>(file_.get());
  if (!impl || !impl->stream.good()) return false;

  // Read the 8-byte length.
  uint64_t length = 0;
  impl->stream.read(reinterpret_cast<char*>(&length), 8);
  if (impl->stream.gcount() != 8) return false;

  // Skip 4-byte length CRC.
  impl->stream.seekg(4, std::ios::cur);

  // Read payload.
  record_.resize(length);
  impl->stream.read(record_.data(), static_cast<std::streamsize>(length));
  if (static_cast<uint64_t>(impl->stream.gcount()) != length) return false;

  // Skip 4-byte payload CRC.
  impl->stream.seekg(4, std::ios::cur);

  offset_ += 8 + 4 + length + 4;
  return true;
}

void TFRecordReader::Close() {
  if (file_) static_cast<TFRecordReaderImpl*>(file_.get())->stream.close();
}

}  // namespace nucleus
