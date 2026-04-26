#include "deepvariant/native/realigner_native.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "deepvariant/realigner/debruijn_graph.h"
#include "deepvariant/realigner/fast_pass_aligner.h"
#include "deepvariant/realigner/window_selector.h"
#include "absl/log/log.h"
#include "third_party/nucleus/protos/range.pb.h"
#include "third_party/nucleus/util/proto_ptr.h"
#include "third_party/nucleus/util/utils.h"

namespace deepvariant {

namespace {

using ::learning::genomics::deepvariant::AlleleCounter;
using ::learning::genomics::deepvariant::DeBruijnGraph;
using ::learning::genomics::deepvariant::FastPassAligner;
using ::learning::genomics::deepvariant::RealignerOptions;
using ::learning::genomics::deepvariant::VariantReadsWindowSelectorCandidates;

constexpr int kRefAlignMargin = 20;

// Container of a candidate window + assigned reads — mirror of
// deepvariant/realigner/realigner.py:AssemblyRegion.
struct AssemblyRegion {
  nucleus::genomics::v1::Range region;
  std::vector<std::string> haplotypes;
  std::vector<int> read_indices;  // indices into the input reads vector
};

// Merge candidate positions into windows of width 2 × min_windows_distance.
// Mirror of window_selector._candidates_to_windows. Only positions with
// `count` in [min_supporting_reads, max_supporting_reads] count as
// "candidates" — without that range filter every read mismatch becomes a
// window seed and the whole region collapses into one >max_window_size
// window that gets discarded.
std::vector<nucleus::genomics::v1::Range> CandidatesToWindows(
    const std::vector<int>& candidate_counts,
    int region_start_pos, const std::string& chrom,
    int min_windows_distance, int max_window_size,
    int min_supporting_reads, int max_supporting_reads) {
  std::vector<nucleus::genomics::v1::Range> windows;
  int start_pos = -1, end_pos = -1;
  auto add_window = [&](int s, int e) {
    nucleus::genomics::v1::Range r;
    r.set_reference_name(chrom);
    r.set_start(std::max(0, s - min_windows_distance));
    r.set_end(e + min_windows_distance);
    if (r.end() - r.start() <= max_window_size) {
      windows.push_back(std::move(r));
    }
  };
  for (int i = 0; i < static_cast<int>(candidate_counts.size()); ++i) {
    const int c = candidate_counts[i];
    if (c < min_supporting_reads || c > max_supporting_reads) continue;
    const int pos = region_start_pos + i;
    if (start_pos == -1) {
      start_pos = end_pos = pos;
    } else if (pos > end_pos + 2 * min_windows_distance) {
      add_window(start_pos, end_pos);
      start_pos = end_pos = pos;
    } else {
      end_pos = pos;
    }
  }
  if (start_pos != -1) add_window(start_pos, end_pos);
  return windows;
}

}  // namespace

RealignerOptions DefaultRealignerOptions() {
  RealignerOptions opts;
  // Window selector — defaults from realigner.py.
  auto* ws = opts.mutable_ws_config();
  ws->set_min_num_supporting_reads(2);
  ws->set_max_num_supporting_reads(300);
  ws->set_min_mapq(20);
  ws->set_min_base_quality(20);
  ws->set_min_windows_distance(80);
  ws->set_max_window_size(1000);
  // De-Bruijn graph — defaults from realigner.py.
  auto* dbg = opts.mutable_dbg_config();
  dbg->set_min_k(10);
  dbg->set_max_k(101);
  dbg->set_step_k(1);
  dbg->set_min_mapq(14);
  dbg->set_min_base_quality(15);
  dbg->set_min_edge_weight(2);
  dbg->set_max_num_paths(256);
  // Aligner — defaults.
  auto* aln = opts.mutable_aln_config();
  aln->set_match(4);
  aln->set_mismatch(6);
  aln->set_gap_open(8);
  aln->set_gap_extend(2);
  aln->set_k(23);
  aln->set_error_rate(0.01);
  aln->set_max_num_of_mismatches(2);
  aln->set_realignment_similarity_threshold(0.16934);
  aln->set_kmer_size(32);
  return opts;
}

std::vector<nucleus::genomics::v1::Read> RealignReadsForRegion(
    const std::vector<nucleus::genomics::v1::Read>& reads,
    const nucleus::genomics::v1::Range& region,
    const AlleleCounter& counter,
    const nucleus::GenomeReference& ref_reader,
    const RealignerOptions& options) {
  if (reads.empty()) return reads;

  // ── Step 1: candidate counts via WindowSelector ──────────────────────────
  std::vector<int> counts =
      VariantReadsWindowSelectorCandidates(counter, options.ws_config());

  // ── Step 2: merge counts into windows ────────────────────────────────────
  auto windows = CandidatesToWindows(
      counts, static_cast<int>(region.start()), region.reference_name(),
      options.ws_config().min_windows_distance(),
      options.ws_config().max_window_size(),
      options.ws_config().min_num_supporting_reads(),
      options.ws_config().max_num_supporting_reads());

  LOG(INFO) << "  realigner: " << windows.size() << " candidate windows in "
            << region.reference_name() << ":" << region.start() << "-"
            << region.end()
            << " (positions with non-zero counts: "
            << std::count_if(counts.begin(), counts.end(),
                             [](int c) { return c > 0; })
            << ")";
  if (windows.empty()) return reads;

  // ── Step 3: build DeBruijn graphs → assembled regions ────────────────────
  std::vector<AssemblyRegion> assembled;
  // We need ConstProtoPtr<Read> for DeBruijnGraph::Build.
  std::vector<nucleus::ConstProtoPtr<const nucleus::genomics::v1::Read>>
      read_ptrs;
  read_ptrs.reserve(reads.size());
  for (const auto& r : reads) {
    read_ptrs.push_back(
        nucleus::ConstProtoPtr<const nucleus::genomics::v1::Read>(&r));
  }

  for (const auto& window : windows) {
    auto ref_or = ref_reader.GetBases(window);
    if (!ref_or.ok()) continue;
    const std::string ref_bases = ref_or.ValueOrDie();

    // Filter reads overlapping the window (DeBruijnGraph filters internally
    // by mapq/base_quality from dbg_config).
    std::vector<nucleus::ConstProtoPtr<const nucleus::genomics::v1::Read>>
        win_reads;
    for (size_t i = 0; i < reads.size(); ++i) {
      if (nucleus::ReadOverlapsRegion(reads[i], window)) {
        win_reads.push_back(read_ptrs[i]);
      }
    }
    if (win_reads.empty()) continue;

    // Build the graph.
    std::vector<nucleus::ConstProtoPtr<const nucleus::genomics::v1::Read>>
        win_reads_copy = win_reads;
    auto graph = DeBruijnGraph::Build(ref_bases, win_reads_copy,
                                       options.dbg_config());
    std::vector<std::string> haplotypes;
    if (graph) {
      haplotypes = graph->CandidateHaplotypes();
    }
    if (haplotypes.empty() ||
        (haplotypes.size() == 1 && haplotypes[0] == ref_bases)) {
      continue;  // Nothing to realign in this window.
    }

    AssemblyRegion ar;
    ar.region = window;
    ar.haplotypes = std::move(haplotypes);
    assembled.push_back(std::move(ar));
  }

  LOG(INFO) << "  realigner: " << assembled.size()
            << " assembled regions in "
            << region.reference_name() << ":" << region.start() << "-"
            << region.end();
  if (assembled.empty()) return reads;

  // ── Step 4: assign reads to assembled regions (first-overlap wins) ───────
  std::vector<bool> read_assigned(reads.size(), false);
  for (auto& ar : assembled) {
    for (size_t i = 0; i < reads.size(); ++i) {
      if (read_assigned[i]) continue;
      if (nucleus::ReadOverlapsRegion(reads[i], ar.region)) {
        ar.read_indices.push_back(static_cast<int>(i));
        read_assigned[i] = true;
      }
    }
  }

  // Start with the unassigned reads (they pass through unchanged).
  std::vector<nucleus::genomics::v1::Read> out;
  out.reserve(reads.size());
  for (size_t i = 0; i < reads.size(); ++i) {
    if (!read_assigned[i]) out.push_back(reads[i]);
  }

  // ── Step 5: realign each assembled region's reads ────────────────────────
  for (const auto& ar : assembled) {
    if (ar.read_indices.empty()) continue;

    const std::string& chrom = ar.region.reference_name();
    auto contig_or = ref_reader.Contig(chrom);
    if (!contig_or.ok()) continue;
    const int64_t contig_n_bases = contig_or.ValueOrDie()->n_bases();

    const int64_t ref_start =
        std::max<int64_t>(0, ar.region.start() - kRefAlignMargin);
    const int64_t ref_end =
        std::min<int64_t>(contig_n_bases, ar.region.end() + kRefAlignMargin);
    if (ref_end <= ar.region.end()) {
      // Can't extend; pass these reads through unchanged.
      for (int idx : ar.read_indices) out.push_back(reads[idx]);
      continue;
    }

    nucleus::genomics::v1::Range pre_range, win_range, suf_range;
    pre_range.set_reference_name(chrom);
    pre_range.set_start(ref_start);
    pre_range.set_end(ar.region.start());
    win_range = ar.region;
    suf_range.set_reference_name(chrom);
    suf_range.set_start(ar.region.end());
    suf_range.set_end(ref_end);

    auto ref_pre_or = ref_reader.GetBases(pre_range);
    auto ref_win_or = ref_reader.GetBases(win_range);
    auto ref_suf_or = ref_reader.GetBases(suf_range);
    if (!ref_pre_or.ok() || !ref_win_or.ok() || !ref_suf_or.ok()) {
      for (int idx : ar.read_indices) out.push_back(reads[idx]);
      continue;
    }
    const std::string ref_pre = ref_pre_or.ValueOrDie();
    const std::string ref_win = ref_win_or.ValueOrDie();
    const std::string ref_suf = ref_suf_or.ValueOrDie();
    const std::string ref_seq = ref_pre + ref_win + ref_suf;

    // Build the per-region read vector for the aligner.
    std::vector<nucleus::genomics::v1::Read> region_reads;
    region_reads.reserve(ar.read_indices.size());
    for (int idx : ar.read_indices) region_reads.push_back(reads[idx]);

    FastPassAligner aligner;
    auto aln_cfg = options.aln_config();
    aln_cfg.set_read_size(static_cast<int>(
        region_reads[0].aligned_sequence().size()));
    aln_cfg.set_force_alignment(false);
    aligner.set_options(aln_cfg);
    aligner.set_reference(ref_seq);
    aligner.set_ref_start(chrom, static_cast<uint64_t>(ref_start));
    aligner.set_ref_prefix_len(static_cast<int>(ref_pre.size()));
    aligner.set_ref_suffix_len(static_cast<int>(ref_suf.size()));
    std::vector<std::string> haplotypes_padded;
    haplotypes_padded.reserve(ar.haplotypes.size());
    for (const auto& hap : ar.haplotypes) {
      haplotypes_padded.push_back(ref_pre + hap + ref_suf);
    }
    aligner.set_haplotypes(haplotypes_padded);

    auto realigned = aligner.AlignReads(absl::MakeConstSpan(region_reads));
    if (realigned && !realigned->empty()) {
      for (auto& r : *realigned) out.push_back(std::move(r));
    } else {
      // Fallback to original reads if alignment failed.
      for (int idx : ar.read_indices) out.push_back(reads[idx]);
    }
  }

  return out;
}

}  // namespace deepvariant
