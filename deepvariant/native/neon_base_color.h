// neon_base_color.h — NEON-accelerated base→color byte mapping (A2.1).
//
// Phase 7 / locked-plan A2.1 deliverable. Replaces the per-byte switch in
// upstream's BaseColor() with a 256-entry LUT (built once from PileupImage
// options) plus a NEON 16-byte chunk-fill via `vqtbl4q_u8`.
//
// Bit-equivalence vs upstream is the gating contract — see
// microtest_neon_base_color.cc for the byte-identity proof on all 256
// possible input bytes.
//
// NOT YET WIRED into the production make_examples pipeline. Wiring is
// staged for the next session, jointly with A2.2 (NEON CIGAR walk) so we
// can land both behind one upstream-divergence diff. This header ships as
// reusable infrastructure, validated.
//
// Algorithmic guarantee:
//   FillBaseColorScalar(out, in, n)     ≡   FillBaseColorNeon(out, in, n)
//   for every n ∈ [0, ∞) and every byte stream `in`. The NEON fast-path
//   triggers for n ≥ 16; trailing bytes use the scalar tail. Both paths
//   read the same `BaseColorTable256` so reproduction is by construction.
//
// References:
//   - Upstream BaseColor switch:
//       deepvariant/channels/read_base_channel.cc:56-72
//       deepvariant/pileup_channel_lib.cc:327-344
//   - PileupImageOptions field defaults:
//       deepvariant/protos/deepvariant.proto

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define DV_NEON_BASE_COLOR_AVAILABLE 1
#else
#  define DV_NEON_BASE_COLOR_AVAILABLE 0
#endif

namespace deepvariant {
namespace neon_base_color {

// Mirror of upstream's BaseColor switch parameters. The four offsets +
// stride combination produces the four non-zero outputs ('A','C','G','T');
// every other byte (lowercase, 'N', whitespace, ASCII control, …) maps to
// 0, matching upstream's `default: return 0;` arm exactly.
struct ColorParams {
  uint8_t base_color_offset_a_and_g;
  uint8_t base_color_offset_t_and_c;
  uint8_t base_color_stride;
};

// Build the 256-entry lookup table indexed by the raw base byte. Cheap
// (256 stores) and only done once per PileupImageEncoder instance.
inline void BuildBaseColorTable256(const ColorParams& p,
                                   uint8_t out[256]) {
  for (int i = 0; i < 256; ++i) out[i] = 0;
  out[(unsigned char)'A'] =
      static_cast<uint8_t>(p.base_color_offset_a_and_g + p.base_color_stride * 3);
  out[(unsigned char)'G'] =
      static_cast<uint8_t>(p.base_color_offset_a_and_g + p.base_color_stride * 2);
  out[(unsigned char)'T'] =
      static_cast<uint8_t>(p.base_color_offset_t_and_c + p.base_color_stride * 1);
  out[(unsigned char)'C'] =
      static_cast<uint8_t>(p.base_color_offset_t_and_c + p.base_color_stride * 0);
}

// Scalar reference path. Always available, always bit-equivalent to
// upstream's switch for the same params. Use this in the microtest as the
// ground truth.
inline void FillBaseColorScalar(uint8_t* out, const char* in, size_t n,
                                const uint8_t table[256]) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = table[(unsigned char)in[i]];
  }
}

// NEON fast-path. Processes 16 bytes per iteration via `vqtbl4q_u8`,
// which performs a 64-byte parallel table lookup. Since the LUT is
// 256-entry but real bases only span ASCII 'A'..'T' (0x41..0x54), we
// shift the input by -0x40 ('@'=0x40) and clamp to [0..63], then index a
// 64-byte table (LUT[0x40..0x7F]). Out-of-range bytes map to 0 (matching
// upstream's `default: 0`) because LUT[0..63] is built to cover only
// {'A','C','G','T'} with everything else 0, and out-of-range queries on
// vqtbl4q_u8 return 0 by ARM spec.
//
// Tail < 16 falls through to the scalar path.
inline void FillBaseColorNeon(uint8_t* out, const char* in, size_t n,
                              const uint8_t table[256]) {
#if DV_NEON_BASE_COLOR_AVAILABLE
  size_t i = 0;
  if (n >= 16) {
    // Pack the [0x40..0x7F] window of `table` into a 64-byte vector.
    // ASCII printable letters live entirely in this range so for any
    // sequencing-grade input this captures all the relevant LUT entries.
    // (Bytes outside [0x40..0x7F] — control, digits, lowercase — all map
    // to 0 in the original 256-LUT, matching this 64-byte window's
    // implicit zeros for indices ≥64 returned by vqtbl4q_u8.)
    uint8x16x4_t tbl;
    tbl.val[0] = vld1q_u8(&table[0x40]);
    tbl.val[1] = vld1q_u8(&table[0x50]);
    tbl.val[2] = vld1q_u8(&table[0x60]);
    tbl.val[3] = vld1q_u8(&table[0x70]);
    const uint8x16_t bias = vdupq_n_u8(0x40);
    for (; i + 16 <= n; i += 16) {
      uint8x16_t bytes = vld1q_u8(reinterpret_cast<const uint8_t*>(in + i));
      // Subtract 0x40 so 'A'(0x41) → 1, 'T'(0x54) → 0x14. Bytes < 0x40
      // wrap to ≥ 0xC0 via uint8 underflow → vqtbl4q returns 0.
      uint8x16_t idx = vsubq_u8(bytes, bias);
      uint8x16_t res = vqtbl4q_u8(tbl, idx);
      vst1q_u8(out + i, res);
    }
  }
  // Tail.
  for (; i < n; ++i) {
    out[i] = table[(unsigned char)in[i]];
  }
#else
  FillBaseColorScalar(out, in, n, table);
#endif
}

}  // namespace neon_base_color
}  // namespace deepvariant
