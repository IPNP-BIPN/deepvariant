/*
 * Copyright 2025 Google LLC.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Sex-chromosome haploid-calling support shared by the native make_examples
// (gVCF reference confidence) and postprocess (genotype correction) stages.
// Mirrors upstream's --haploid_contigs / --par_regions_bed handling in
// variant_caller.py and postprocess_variants.py: a position on a haploid
// contig that does not overlap a pseudo-autosomal region is called haploid.

#ifndef LEARNING_GENOMICS_DEEPVARIANT_NATIVE_HAPLOID_REGIONS_H_
#define LEARNING_GENOMICS_DEEPVARIANT_NATIVE_HAPLOID_REGIONS_H_

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace deepvariant {

// Pseudo-autosomal regions loaded from a BED file: contig -> 0-based
// half-open [start, end) intervals. A position on a haploid contig that
// overlaps one of these stays diploid.
struct ParRegions {
  std::map<std::string, std::vector<std::pair<int64_t, int64_t>>> by_contig;

  bool Overlaps(const std::string& chrom, int64_t start, int64_t end) const {
    auto it = by_contig.find(chrom);
    if (it == by_contig.end()) return false;
    for (const auto& [s, e] : it->second) {
      if (start < e && end > s) return true;
    }
    return false;
  }
};

// Parse the whitespace/comma-separated --haploid_contigs flag into a set,
// mirroring upstream's split on ',' then on whitespace.
inline std::set<std::string> ParseHaploidContigs(const std::string& flag) {
  std::set<std::string> out;
  for (absl::string_view tok :
       absl::StrSplit(flag, absl::ByAnyChar(", \t\r\n"), absl::SkipEmpty())) {
    out.emplace(tok);
  }
  return out;
}

// Load a PAR BED (chrom \t start \t end, 0-based half-open). Returns false on
// an unreadable file or a malformed line so a typo'd path fails fast rather
// than silently disabling the exemption.
inline bool LoadParRegions(const std::string& path, ParRegions* out,
                           std::string* err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    *err = absl::StrCat("cannot open --par_regions_bed: ", path);
    return false;
  }
  // This is a plaintext parser; upstream's htslib BedReader also accepts
  // bgzipped BED. Detect the gzip magic (0x1f 0x8b) and fail with a clear
  // message rather than mis-parsing binary as malformed lines.
  const int b0 = f.peek();
  if (b0 == 0x1f) {
    f.get();
    const int b1 = f.peek();
    f.unget();
    if (b1 == 0x8b) {
      *err = absl::StrCat("--par_regions_bed appears gzipped (", path,
                          "); provide a plaintext BED");
      return false;
    }
  }
  std::string line;
  int lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#') continue;
    if (line.rfind("track", 0) == 0 || line.rfind("browser", 0) == 0) continue;
    std::vector<absl::string_view> cols =
        absl::StrSplit(line, absl::ByAnyChar("\t "), absl::SkipEmpty());
    int64_t start, end;
    if (cols.size() < 3 || !absl::SimpleAtoi(cols[1], &start) ||
        !absl::SimpleAtoi(cols[2], &end) || start < 0 || end < start) {
      *err = absl::StrCat("malformed BED line ", lineno, " in ", path, ": ",
                          line);
      return false;
    }
    out->by_contig[std::string(cols[0])].emplace_back(start, end);
  }
  return true;
}

// True when [start, end) on `chrom` should be called haploid: the contig is in
// `haploid_contigs` and the interval does not overlap a PAR region. Mirror of
// `reference_name in haploid_contigs and not par_regions.overlaps(...)`.
inline bool IsHaploidPosition(const std::string& chrom, int64_t start,
                              int64_t end,
                              const std::set<std::string>& haploid_contigs,
                              const ParRegions& par_regions) {
  if (haploid_contigs.empty()) return false;
  if (!haploid_contigs.count(chrom)) return false;
  return !par_regions.Overlaps(chrom, start, end);
}

// Zero every heterozygous genotype's probability and renormalize, forcing a
// haploid call. Probabilities are in PL order F(j/k)=k*(k+1)/2+j (j<=k); het
// genotypes are those with j != k. Mirror of
// postprocess_variants.py:correct_nonautosome_probabilities.
inline void CorrectNonautosomeProbabilities(std::vector<double>* like,
                                            int n_alts) {
  for (int k = 0; k <= n_alts; ++k) {
    for (int j = 0; j <= k; ++j) {
      if (j != k) (*like)[k * (k + 1) / 2 + j] = 0.0;
    }
  }
  double sum = 0.0;
  for (double v : *like) sum += v;
  if (sum <= 0.0) sum = 1.0;
  for (double& v : *like) v /= sum;
}

}  // namespace deepvariant

#endif  // LEARNING_GENOMICS_DEEPVARIANT_NATIVE_HAPLOID_REGIONS_H_
