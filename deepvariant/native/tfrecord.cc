// TFRecord reader/writer for deepvariant native runtime.
// See tfrecord.h for the format description.

#include "deepvariant/native/tfrecord.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

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
        if (static_cast<uint64_t>(s.gcount()) != length) {
          // BUG FIX (2026-05-10): the previous `return false` here would
          // ABANDON all remaining shards in a multi-shard read whenever
          // the LAST record of any shard was truncated. On a 14-shard
          // WG run this caused 13/14 shards (~95 % of examples) to be
          // silently dropped: call_variants only saw 69k of 954k
          // examples → 947k PASS calls missing in the final VCF.
          //
          // Truncation cause: upstream's ExamplesGenerator destructor
          // closes the writer without an explicit flush — the last
          // partial-buffer write (1 record per shard, ≈10-150 KB out
          // of 1 MiB buffer) is dropped on close.
          //
          // Fix: treat partial-payload same as EOF — fall through to
          // shard-advance code. Loses the 1 truncated record per shard
          // (unrecoverable since it was never written to disk) but
          // preserves all following shards. WG impact: 14 lost records
          // out of 954k = 0.0015 % vs 100 % loss before the fix.
          // Fall through to shard-advance code below.
        } else {
          s.seekg(4, std::ios::cur);  // skip payload CRC
          offset_ += 8 + 4 + length + 4;
          return true;
        }
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
//
// Implementation note (2026-05-01): we used to back this with
// std::ofstream, which buffers writes in a userspace buffer and lets
// the kernel buffer dirty pages indefinitely. On macOS that triggers
// Jetsam after ~137 GB of dirty file-backed memory in a 24h window,
// killing our process mid-WG run. Switched to a raw POSIX fd with
// F_NOCACHE so writes go straight to the disk device without
// accumulating in the kernel page cache. We keep a small userspace
// buffer (kBufBytes) so each fd write is large enough that the SSD
// can actually batch them; no perf regression observed on chr20.

namespace {
constexpr size_t kBufBytes = 1 << 20;  // 1 MiB write coalescing buffer
}

struct TFRecordWriter::Impl {
  int fd = -1;
  std::vector<char> buf;
  size_t buf_used = 0;
  bool ok = false;

  explicit Impl(const std::string& path) : buf(kBufBytes) {
    fd = ::open(path.c_str(),
                O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    // F_NOCACHE: bypass the unified buffer cache. Writes go straight
    // to disk; pages are NOT marked dirty in the kernel's accounting,
    // so Jetsam doesn't accumulate quota. Only available on macOS.
    ::fcntl(fd, F_NOCACHE, 1);
    // Pre-allocate a hint to the FS for sequential write.
    fcntl(fd, F_RDADVISE, 0);  // best-effort; ignored if unsupported
    ok = true;
  }

  ~Impl() {
    FlushBuf();
    if (fd >= 0) ::close(fd);
  }

  bool FlushBuf() {
    if (!ok || buf_used == 0) return ok;
    const char* p = buf.data();
    size_t left = buf_used;
    while (left > 0) {
      ssize_t n = ::write(fd, p, left);
      if (n <= 0) { ok = false; return false; }
      p += n;
      left -= static_cast<size_t>(n);
    }
    buf_used = 0;
    return true;
  }

  bool Append(const char* data, size_t n) {
    if (!ok) return false;
    while (n > 0) {
      const size_t room = buf.size() - buf_used;
      const size_t take = std::min(n, room);
      std::memcpy(buf.data() + buf_used, data, take);
      buf_used += take;
      data += take;
      n -= take;
      if (buf_used == buf.size()) {
        if (!FlushBuf()) return false;
      }
    }
    return true;
  }
};

TFRecordWriter::TFRecordWriter() = default;
TFRecordWriter::~TFRecordWriter() = default;

std::unique_ptr<TFRecordWriter> TFRecordWriter::New(
    const std::string& path, const std::string& /*compression_type*/) {
  auto impl = std::make_unique<Impl>(path);
  if (!impl->ok) return nullptr;
  auto w = std::unique_ptr<TFRecordWriter>(new TFRecordWriter());
  w->impl_ = std::move(impl);
  return w;
}

bool TFRecordWriter::WriteRecord(const std::string& payload) {
  if (!impl_ || !impl_->ok) return false;
  uint64_t len = payload.size();
  uint32_t len_crc =
      MaskedCrc32c(reinterpret_cast<const char*>(&len), sizeof(len));
  uint32_t data_crc = MaskedCrc32c(payload.data(), len);
  if (!impl_->Append(reinterpret_cast<const char*>(&len), 8)) return false;
  if (!impl_->Append(reinterpret_cast<const char*>(&len_crc), 4)) return false;
  if (!impl_->Append(payload.data(), len)) return false;
  if (!impl_->Append(reinterpret_cast<const char*>(&data_crc), 4)) return false;
  return true;
}

bool TFRecordWriter::Flush() {
  if (!impl_) return false;
  return impl_->FlushBuf();
}

bool TFRecordWriter::Close() {
  if (!impl_) return true;
  bool ok = impl_->FlushBuf();
  if (impl_->fd >= 0) {
    ::close(impl_->fd);
    impl_->fd = -1;
  }
  return ok;
}

}  // namespace deepvariant
