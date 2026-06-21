// TFRecord reader/writer for the deepvariant native runtime.
// Binary format: [uint64_le length][uint32_le masked_crc32c(len)]
//                [bytes payload][uint32_le masked_crc32c(payload)]
// No TF types in this interface — pure C++ with std::string.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace deepvariant {

// Read TFRecord files sequentially.  One instance is NOT thread-safe.
class TFRecordReader {
 public:
  // Valid compression_type: "" (none). GZIP/ZLIB not supported.
  static std::unique_ptr<TFRecordReader> New(
      const std::string& path, const std::string& compression_type = "");

  ~TFRecordReader();

  // Advance to next record; returns true if a record was read.
  bool GetNext();

  // Current record payload (only valid after a successful GetNext()).
  const std::string& record() const { return record_; }

  void Close();

  TFRecordReader(const TFRecordReader&) = delete;
  TFRecordReader& operator=(const TFRecordReader&) = delete;

 private:
  TFRecordReader();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string record_;
  uint64_t offset_ = 0;
};

// Write TFRecord files.  One instance is NOT thread-safe.
class TFRecordWriter {
 public:
  // Valid compression_type: "" (none).
  static std::unique_ptr<TFRecordWriter> New(
      const std::string& path, const std::string& compression_type = "");

  ~TFRecordWriter();

  bool WriteRecord(const std::string& payload);
  bool Flush();
  bool Close();

  TFRecordWriter(const TFRecordWriter&) = delete;
  TFRecordWriter& operator=(const TFRecordWriter&) = delete;

 private:
  TFRecordWriter();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace deepvariant
