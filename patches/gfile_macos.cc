// POSIX replacement for third_party/nucleus/io/gfile.cc.
// Drop-in implementation of nucleus::Exists, nucleus::Glob,
// nucleus::ReadableFile, nucleus::WritableFile using standard C++17 / POSIX
// — zero dependency on TensorFlow.

#include "third_party/nucleus/io/gfile.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <glob.h>
#include <memory>
#include <string>
#include <vector>

namespace nucleus {

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

bool Exists(const std::string& filename) {
  return std::filesystem::exists(filename);
}

std::vector<std::string> Glob(const std::string& pattern) {
  std::vector<std::string> results;
  glob_t g{};
  if (glob(pattern.c_str(), GLOB_TILDE, nullptr, &g) == 0) {
    for (size_t i = 0; i < g.gl_pathc; ++i)
      results.emplace_back(g.gl_pathv[i]);
  }
  globfree(&g);
  return results;
}

// ---------------------------------------------------------------------------
// ReadableFile
// ---------------------------------------------------------------------------

struct ReadableFileImpl {
  std::ifstream stream;
  explicit ReadableFileImpl(const std::string& path) : stream(path) {}
};

ReadableFile::ReadableFile() = default;

ReadableFile::~ReadableFile() = default;

// static
std::unique_ptr<ReadableFile> ReadableFile::New(const std::string& filename) {
  auto f = std::unique_ptr<ReadableFile>(new ReadableFile());
  auto impl = std::make_unique<ReadableFileImpl>(filename);
  if (!impl->stream.is_open()) return nullptr;
  f->stream_ = std::move(impl);
  return f;
}

bool ReadableFile::Readline(std::string* s) {
  auto* impl = static_cast<ReadableFileImpl*>(stream_.get());
  return static_cast<bool>(std::getline(impl->stream, *s));
}

void ReadableFile::Close() {
  if (stream_) static_cast<ReadableFileImpl*>(stream_.get())->stream.close();
}

// ---------------------------------------------------------------------------
// WritableFile
// ---------------------------------------------------------------------------

struct WritableFileImpl {
  std::ofstream stream;
  explicit WritableFileImpl(const std::string& path) : stream(path) {}
};

WritableFile::WritableFile() = default;

WritableFile::~WritableFile() = default;

// static
std::unique_ptr<WritableFile> WritableFile::New(const std::string& filename) {
  auto f = std::unique_ptr<WritableFile>(new WritableFile());
  auto impl = std::make_unique<WritableFileImpl>(filename);
  if (!impl->stream.is_open()) return nullptr;
  f->file_ = std::move(impl);
  return f;
}

bool WritableFile::Write(const std::string& s) {
  auto* impl = static_cast<WritableFileImpl*>(file_.get());
  impl->stream.write(s.data(), static_cast<std::streamsize>(s.size()));
  return impl->stream.good();
}

void WritableFile::Close() {
  if (file_) static_cast<WritableFileImpl*>(file_.get())->stream.close();
}

}  // namespace nucleus
