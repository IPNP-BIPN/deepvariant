// Phase 7 / locked-plan A2.1 microtest — verify FillBaseColorNeon
// produces byte-identical output to FillBaseColorScalar for every byte
// in [0..255] and across realistic pileup-row lengths.
//
// Gating contract (must hold or A2.1 is unsafe to wire into production):
//   FillBaseColorScalar(out_a, in, n) == FillBaseColorNeon(out_b, in, n)
//   for every (n, byte stream `in`, params).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "deepvariant/native/neon_base_color.h"

using deepvariant::neon_base_color::BuildBaseColorTable256;
using deepvariant::neon_base_color::ColorParams;
using deepvariant::neon_base_color::FillBaseColorNeon;
using deepvariant::neon_base_color::FillBaseColorScalar;

namespace {

// Upstream's BaseColor switch transcribed verbatim — used as the
// independent ground truth (not the LUT, which both NEON and scalar
// paths share).
inline uint8_t BaseColorUpstream(char base, const ColorParams& p) {
  switch (base) {
    case 'A':
      return static_cast<uint8_t>(p.base_color_offset_a_and_g +
                                  p.base_color_stride * 3);
    case 'G':
      return static_cast<uint8_t>(p.base_color_offset_a_and_g +
                                  p.base_color_stride * 2);
    case 'T':
      return static_cast<uint8_t>(p.base_color_offset_t_and_c +
                                  p.base_color_stride * 1);
    case 'C':
      return static_cast<uint8_t>(p.base_color_offset_t_and_c +
                                  p.base_color_stride * 0);
    default:
      return 0;
  }
}

}  // namespace

int main() {
  int n_fail = 0;

  // Default PileupImage parameters from deepvariant/protos/deepvariant.proto:
  //   base_color_offset_a_and_g = 40
  //   base_color_offset_t_and_c = 30
  //   base_color_stride         = 70
  ColorParams params{40, 30, 70};
  uint8_t table[256];
  BuildBaseColorTable256(params, table);

  // Test 1: LUT itself byte-matches upstream's switch on every byte 0..255.
  {
    std::printf("Test 1: LUT byte-match vs upstream switch on all 256 bytes\n");
    int n_diff = 0;
    for (int b = 0; b < 256; ++b) {
      uint8_t up = BaseColorUpstream((char)b, params);
      if (up != table[b]) {
        std::printf("  byte=0x%02x ('%c'): upstream=%u table=%u\n",
                    b, (b >= 32 && b < 127) ? b : '?', up, table[b]);
        ++n_diff;
      }
    }
    if (n_diff == 0) std::printf("  -> 256/256 PASS\n");
    else { std::printf("  -> %d FAIL\n", n_diff); ++n_fail; }
  }

  // Test 2: NEON vs scalar path, pileup-realistic input ('A','C','G','T','N').
  {
    std::printf("Test 2: NEON vs scalar on ACGT/N strings (length 0..1024)\n");
    std::mt19937 rng(0xDEADBEEFu);
    static const char alphabet[] = "ACGTN";
    int n_diff = 0;
    for (size_t n = 0; n <= 1024; ++n) {
      std::vector<char> in(n);
      std::vector<uint8_t> out_scalar(n + 16, 0xAB);
      std::vector<uint8_t> out_neon(n + 16, 0xCD);
      for (size_t i = 0; i < n; ++i) in[i] = alphabet[rng() % 5];
      FillBaseColorScalar(out_scalar.data(), in.data(), n, table);
      FillBaseColorNeon(out_neon.data(), in.data(), n, table);
      // Body must match.
      if (std::memcmp(out_scalar.data(), out_neon.data(), n) != 0) {
        for (size_t i = 0; i < n; ++i) {
          if (out_scalar[i] != out_neon[i]) {
            std::printf("  n=%zu i=%zu in='%c' scalar=%u neon=%u\n",
                        n, i, in[i], out_scalar[i], out_neon[i]);
            ++n_diff;
            if (n_diff > 10) break;
          }
        }
      }
      // Tail-overshoot guard: NEON must not write past `out + n`.
      for (size_t i = n; i < n + 16; ++i) {
        if (out_neon[i] != 0xCD) {
          std::printf("  n=%zu OVERSHOOT at +%zu (got %u)\n",
                      n, i - n, out_neon[i]);
          ++n_diff;
        }
      }
      if (n_diff > 100) break;
    }
    if (n_diff == 0) std::printf("  -> 1025/1025 lengths PASS, no overshoot\n");
    else { std::printf("  -> %d FAIL\n", n_diff); ++n_fail; }
  }

  // Test 3: NEON vs scalar on adversarial input — every possible byte at
  // every possible alignment within a 16-byte chunk.
  {
    std::printf("Test 3: NEON vs scalar on all-byte input (256-byte block)\n");
    std::vector<char> in(256);
    for (int i = 0; i < 256; ++i) in[i] = (char)i;
    std::vector<uint8_t> out_scalar(256);
    std::vector<uint8_t> out_neon(256);
    FillBaseColorScalar(out_scalar.data(), in.data(), 256, table);
    FillBaseColorNeon(out_neon.data(), in.data(), 256, table);
    int n_diff = 0;
    for (int i = 0; i < 256; ++i) {
      if (out_scalar[i] != out_neon[i]) {
        std::printf("  byte=0x%02x scalar=%u neon=%u\n",
                    i, out_scalar[i], out_neon[i]);
        ++n_diff;
      }
    }
    if (n_diff == 0) std::printf("  -> 256/256 PASS\n");
    else { std::printf("  -> %d FAIL\n", n_diff); ++n_fail; }
  }

  // Test 4: alternate params (stride=1, offsets=10/20).
  {
    std::printf("Test 4: alternate ColorParams (stride=1, offsets=10/20)\n");
    ColorParams alt{10, 20, 1};
    uint8_t alt_table[256];
    BuildBaseColorTable256(alt, alt_table);
    std::vector<char> in(257);
    for (int i = 0; i < 257; ++i) in[i] = (char)((i * 73) & 0xFF);
    std::vector<uint8_t> out_scalar(257), out_neon(257);
    FillBaseColorScalar(out_scalar.data(), in.data(), 257, alt_table);
    FillBaseColorNeon(out_neon.data(), in.data(), 257, alt_table);
    int n_diff = 0;
    for (int i = 0; i < 257; ++i) {
      uint8_t up = BaseColorUpstream(in[i], alt);
      if (out_scalar[i] != up || out_neon[i] != up) {
        ++n_diff;
        if (n_diff <= 5)
          std::printf("  i=%d byte=0x%02x up=%u scalar=%u neon=%u\n",
                      i, (unsigned char)in[i], up, out_scalar[i], out_neon[i]);
      }
    }
    if (n_diff == 0) std::printf("  -> 257/257 PASS, alt-params bit-exact\n");
    else { std::printf("  -> %d FAIL\n", n_diff); ++n_fail; }
  }

  // Test 5: throughput microbench — pileup-realistic 221-byte row,
  // amortized over 1 M iterations. Each iteration mutates one input
  // byte to defeat the compiler's invariant-load-store elimination.
  {
    std::printf("Test 5: throughput on 221-byte rows x 1M iter\n");
    constexpr size_t kRowLen = 221;
    constexpr size_t kIter = 1'000'000;
    std::vector<char> in(kRowLen);
    std::vector<uint8_t> out(kRowLen);
    std::mt19937 rng(0xCAFEBABEu);
    static const char alphabet[] = "ACGTN";
    for (size_t i = 0; i < kRowLen; ++i) in[i] = alphabet[rng() % 5];

    auto bench = [&](auto fn, const char* name) {
      // Warm-up.
      uint64_t sink = 0;
      for (int w = 0; w < 1000; ++w) {
        fn(out.data(), in.data(), kRowLen, table);
        sink += out[w & (kRowLen - 1)];
      }
      auto t0 = std::chrono::steady_clock::now();
      for (size_t i = 0; i < kIter; ++i) {
        // Mutate one byte each iter so the compiler cannot hoist.
        in[i & (kRowLen - 1)] = alphabet[i & 3];
        fn(out.data(), in.data(), kRowLen, table);
        sink += out[i & (kRowLen - 1)];
      }
      auto t1 = std::chrono::steady_clock::now();
      double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
      // sink is volatile-printed so the optimizer can't elide it.
      std::printf("  %-12s : %.2f ns/row (sink=%llu)\n",
                  name, ns / kIter, (unsigned long long)sink);
      return ns;
    };
    double s_ns = bench(FillBaseColorScalar, "scalar");
    double n_ns = bench(FillBaseColorNeon, "neon");
    std::printf("  speed-up : %.2fx\n", s_ns / n_ns);
  }

  std::printf("\n%d test%s failed\n", n_fail, n_fail == 1 ? "" : "s");
  return n_fail == 0 ? 0 : 1;
}
