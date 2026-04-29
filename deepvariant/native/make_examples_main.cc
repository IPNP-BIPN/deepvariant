// Native make_examples — calling mode only (no training, no labeling).
// Replaces the Python make_examples_core.py orchestration layer.
//
// Pipeline per region:
//   SamReader.Query → AlleleCounter → VariantCaller → ExamplesGenerator
//
// The heavy C++ implementations (AlleleCounter, VariantCaller, pileup image
// encoding) are fully reused from upstream; only the orchestration is new.

#include "deepvariant/native/make_examples_main.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "deepvariant/allelecounter.h"
#include "deepvariant/make_examples_native.h"
#include "deepvariant/native/numpy_mt19937.h"
#include "deepvariant/native/realigner_native.h"
#include "deepvariant/native/regions.h"
#include "deepvariant/native/small_model_features.h"
#include "deepvariant/native/small_model_inference.h"
#include "deepvariant/native/tfrecord.h"
#include "deepvariant/protos/deepvariant.pb.h"
#include "deepvariant/protos/realigner.pb.h"
#include "deepvariant/variant_calling.h"
#include "deepvariant/variant_calling_multisample.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "third_party/nucleus/io/reference.h"
#include "third_party/nucleus/io/sam_reader.h"
#include "third_party/nucleus/protos/range.pb.h"
#include "third_party/nucleus/protos/reads.pb.h"
#include "third_party/nucleus/protos/reference.pb.h"
#include "third_party/nucleus/protos/struct.pb.h"
#include "third_party/nucleus/util/proto_ptr.h"
#include "third_party/nucleus/util/utils.h"
#include <cmath>

ABSL_FLAG(std::string, reads, "", "BAM/CRAM file with aligned reads.");
ABSL_FLAG(std::string, ref, "", "Reference FASTA (.fai index required).");
// `--examples` is the canonical pipeline filespec — defined in call_variants.
ABSL_DECLARE_FLAG(std::string, examples);
ABSL_FLAG(std::string, regions, "",
          "Whitespace-separated region strings (e.g. 'chr20 chr21:1-1000000')."
          " Empty = all contigs.");
ABSL_FLAG(std::string, exclude_regions, "",
          "Whitespace-separated regions to exclude.");
ABSL_FLAG(int, task_id, 0, "0-based shard index.");
ABSL_FLAG(int, num_shards, 0,
          "Total shards. 0 or 1 means no sharding.");
ABSL_FLAG(std::string, sample_name, "",
          "Sample name (inferred from BAM header if empty).");
// Variant calling thresholds — WGS defaults.
ABSL_FLAG(int, vsc_min_count_snps, 2, "Min supporting read count for SNPs.");
ABSL_FLAG(int, vsc_min_count_indels, 2,
          "Min supporting read count for indels.");
ABSL_FLAG(double, vsc_min_fraction_snps, 0.12,
          "Min allele fraction for SNPs.");
ABSL_FLAG(double, vsc_min_fraction_indels, 0.06,
          "Min allele fraction for indels.");
ABSL_FLAG(int, partition_size, 1000,
          "AlleleCounter partition size (bp per window).");
// Default 5 mirrors upstream's make_examples_options.py
// (`--min_mapping_quality` default = 5). The candidate-emission
// AlleleCounter uses this; the WindowSelector / DBG apply their own
// stricter thresholds (20 / 14).
ABSL_FLAG(int, min_mapping_quality, 5, "Min read mapping quality.");
ABSL_FLAG(int, min_base_quality, 10, "Min base quality.");
// Small model first-pass.
ABSL_FLAG(std::string, small_model, "",
          "Path to the small_model .mlpackage. Empty = no small model "
          "(every candidate goes through the big InceptionV3 model).");
ABSL_FLAG(std::string, small_model_cvo_outfile, "",
          "TFRecord path for CVOs the small model decides directly. "
          "Read by postprocess_variants alongside the big-model CVOs.");
ABSL_FLAG(int, small_model_snp_gq_threshold, 20,
          "Min phred GQ for the small model to commit a SNP call.");
ABSL_FLAG(int, small_model_indel_gq_threshold, 28,
          "Min phred GQ for the small model to commit an indel call.");
ABSL_FLAG(bool, realigner_enabled, false,
          "Enable upstream's realigner (DeBruijnGraph + FastPassAligner) "
          "to recover candidates in indel-rich regions.");
ABSL_FLAG(int, threads, 1,
          "Worker threads inside this process. >1 enables true intra-process "
          "parallelism (one process showing N×100 % CPU). Each worker opens "
          "its own SamReader / IndexedFastaReader / ExamplesGenerator / "
          "SmallModel and writes to a per-thread file; results are "
          "concatenated into the final --examples / --small_model_cvo_outfile "
          "paths after all workers join.");

// ----------------------------------------------------------------------------
// DeepTrio flags (Step 1 — mirrors deeptrio/make_examples.py exactly).
// When --reads_parent1 is set, make_examples runs in trio mode: 3 samples
// (parent1 at index 0, child at index 1, parent2 at index 2; child is the
// MAIN_SAMPLE_INDEX). Each region is processed by 3 AlleleCounters keyed by
// sample_name and fed to multi_sample::VariantCaller. ExamplesGenerator
// emits 3 separate example streams (one per target sample), each rendered
// with the per-sample `order` permutation so the pileup channel-stack
// shows the target sample in slot 1.
// ----------------------------------------------------------------------------
ABSL_FLAG(std::string, reads_parent1, "",
          "Trio mode: BAM/CRAM for parent1. When set, make_examples runs "
          "as DeepTrio (3 samples: parent1, child, parent2; child = main).");
ABSL_FLAG(std::string, reads_parent2, "",
          "Trio mode: BAM/CRAM for parent2.");
ABSL_FLAG(std::string, sample_name_parent1, "",
          "Trio mode: parent1 sample name (inferred from BAM if empty).");
ABSL_FLAG(std::string, sample_name_parent2, "",
          "Trio mode: parent2 sample name (inferred from BAM if empty).");
ABSL_FLAG(int, pileup_image_height_child, 0,
          "Trio mode: pileup image height for the child sample. 0 = default "
          "(100 per upstream dt_constants.PILEUP_DEFAULT_HEIGHT_CHILD).");
ABSL_FLAG(int, pileup_image_height_parent, 0,
          "Trio mode: pileup image height for each parent sample. 0 = default "
          "(100 per upstream dt_constants.PILEUP_DEFAULT_HEIGHT_PARENT).");
ABSL_FLAG(double, downsample_fraction_child, 0.0,
          "Trio mode: downsample fraction applied to child reads (0.0 = none).");
ABSL_FLAG(double, downsample_fraction_parents, 0.0,
          "Trio mode: downsample fraction applied to both parents' reads.");
ABSL_FLAG(std::string, small_model_path_child, "",
          "Trio mode: small_model weights directory for child examples.");
ABSL_FLAG(std::string, small_model_path_parent, "",
          "Trio mode: small_model weights directory for parent examples.");
ABSL_FLAG(bool, skip_parent_calling, false,
          "Trio mode: if true, generate examples for child only "
          "(parents' SampleOptions still populated for joint candidate "
          "generation, but their example output is suppressed).");
ABSL_FLAG(std::string, examples_child, "",
          "Trio mode: examples output path for the child sample. If empty, "
          "the existing --examples flag is used as the child path.");
ABSL_FLAG(std::string, examples_parent1, "",
          "Trio mode: examples output path for the parent1 sample.");
ABSL_FLAG(std::string, examples_parent2, "",
          "Trio mode: examples output path for the parent2 sample.");
ABSL_FLAG(std::string, small_model_cvo_outfile_child, "",
          "Trio mode: small_model CVO output path for child.");
ABSL_FLAG(std::string, small_model_cvo_outfile_parent1, "",
          "Trio mode: small_model CVO output path for parent1.");
ABSL_FLAG(std::string, small_model_cvo_outfile_parent2, "",
          "Trio mode: small_model CVO output path for parent2.");

namespace deepvariant {

using namespace learning::genomics::deepvariant;  // NOLINT

namespace {

// Build the MakeExamplesOptions proto for calling mode from flags.
MakeExamplesOptions BuildOptions(const std::string& sample_name,
                                 int task_id, int num_shards) {
  MakeExamplesOptions opts;

  opts.set_reference_filename(absl::GetFlag(FLAGS_ref));
  opts.set_examples_filename(absl::GetFlag(FLAGS_examples));
  opts.set_task_id(task_id);
  opts.set_num_shards(num_shards);
  opts.set_mode(MakeExamplesOptions::CALLING);
  opts.set_random_seed(609314161);
  // Match upstream `make_examples_options.py`: cap reads per
  // partition (default 1500) so high-coverage regions don't blow up
  // and so our per-region read selection matches Docker's. Reservoir
  // sampling is applied inside the per-region worker loop with a
  // NumPy-compatible RNG (numpy_mt19937.h).
  opts.set_max_reads_per_partition(1500);

  // Read requirements.
  nucleus::genomics::v1::ReadRequirements read_reqs;
  read_reqs.set_min_mapping_quality(absl::GetFlag(FLAGS_min_mapping_quality));
  read_reqs.set_min_base_quality(absl::GetFlag(FLAGS_min_base_quality));
  read_reqs.set_min_base_quality_mode(
      nucleus::genomics::v1::ReadRequirements::ENFORCED_BY_CLIENT);

  // Allele counter options.
  AlleleCounterOptions ac_opts;
  ac_opts.set_partition_size(absl::GetFlag(FLAGS_partition_size));
  *ac_opts.mutable_read_requirements() = read_reqs;
  // Required so AlleleCounter actually retains REF-supporting reads in
  // each AlleleCount.read_alleles map (otherwise the small_model sees
  // num_reads_supports_ref = 0 on every candidate and is biased).
  ac_opts.set_track_ref_reads(true);
  *opts.mutable_allele_counter_options() = ac_opts;

  // Variant caller options.
  VariantCallerOptions vc_opts;
  vc_opts.set_min_count_snps(absl::GetFlag(FLAGS_vsc_min_count_snps));
  vc_opts.set_min_count_indels(absl::GetFlag(FLAGS_vsc_min_count_indels));
  vc_opts.set_min_fraction_snps(absl::GetFlag(FLAGS_vsc_min_fraction_snps));
  vc_opts.set_min_fraction_indels(
      absl::GetFlag(FLAGS_vsc_min_fraction_indels));
  vc_opts.set_p_error(0.001);
  vc_opts.set_max_gq(50);
  vc_opts.set_gq_resolution(1);
  vc_opts.set_ploidy(2);
  vc_opts.set_fraction_reference_sites_to_emit(0.0);
  vc_opts.set_random_seed(1260872234);
  // Required so variant_calling_multisample.cc populates ref_support_ext —
  // without it the small_model sees zero ref-supporting reads on every
  // candidate and predicts hom_ref for everything.
  vc_opts.set_track_ref_reads(true);

  // Pileup image options (WGS defaults).
  PileupImageOptions pic;
  pic.set_reference_band_height(5);
  pic.set_base_color_offset_a_and_g(40);
  pic.set_base_color_offset_t_and_c(30);
  pic.set_base_color_stride(70);
  pic.set_allele_supporting_read_alpha(1.0f);
  pic.set_allele_unsupporting_read_alpha(0.6f);
  pic.set_other_allele_supporting_read_alpha(0.6f);
  pic.set_reference_matching_read_alpha(0.2f);
  pic.set_reference_mismatching_read_alpha(1.0f);
  pic.set_indel_anchoring_base_char("*");
  pic.set_reference_alpha(0.4f);
  pic.set_reference_base_quality(60);
  pic.set_positive_strand_color(70);
  pic.set_negative_strand_color(240);
  pic.set_base_quality_cap(40);
  pic.set_mapping_quality_cap(60);
  pic.set_height(100);
  pic.set_width(221);
  pic.set_read_overlap_buffer_bp(5);
  pic.set_multi_allelic_mode(PileupImageOptions::ADD_HET_ALT_IMAGES);
  pic.set_random_seed(2101079370);
  pic.set_alt_aligned_pileup("none");
  pic.set_types_to_alt_align("indels");
  pic.set_min_non_zero_allele_frequency(0.00001f);
  *pic.mutable_read_requirements() = read_reqs;
  // Default channels for WGS: 6 base + insert_size = 7 (matches model input).
  pic.add_channels("read_base");
  pic.add_channels("base_quality");
  pic.add_channels("mapping_quality");
  pic.add_channels("strand");
  pic.add_channels("read_supports_variant");
  pic.add_channels("base_differs_from_ref");
  pic.add_channels("insert_size");
  pic.set_num_channels(7);
  *opts.mutable_pic_options() = pic;

  // Sample options. Trio mode (--reads_parent1 set) populates 3 samples
  // in upstream order [parent1, child, parent2] (mirrors deeptrio/
  // make_examples.py:trio_samples_from_flags). Single-sample mode keeps
  // the legacy single SampleOptions.
  const std::string parent1_reads = absl::GetFlag(FLAGS_reads_parent1);
  const std::string parent2_reads = absl::GetFlag(FLAGS_reads_parent2);
  const bool trio_mode = !parent1_reads.empty();

  if (trio_mode) {
    // Per upstream dt_constants:
    //   PILEUP_DEFAULT_HEIGHT_CHILD  = 100
    //   PILEUP_DEFAULT_HEIGHT_PARENT = 100
    int child_h  = absl::GetFlag(FLAGS_pileup_image_height_child);
    int parent_h = absl::GetFlag(FLAGS_pileup_image_height_parent);
    if (child_h  <= 0) child_h  = 100;
    if (parent_h <= 0) parent_h = 100;
    const double ds_child   = absl::GetFlag(FLAGS_downsample_fraction_child);
    const double ds_parents = absl::GetFlag(FLAGS_downsample_fraction_parents);
    const std::string p1_name = absl::GetFlag(FLAGS_sample_name_parent1);
    const std::string p2_name = absl::GetFlag(FLAGS_sample_name_parent2);

    auto add_sample = [&](const std::string& role, const std::string& name,
                           const std::string& reads, int height, double ds,
                           std::initializer_list<int> order,
                           bool skip_output, const std::string& small_path) {
      SampleOptions* s = opts.add_sample_options();
      s->set_role(role);
      s->set_name(name);
      if (!reads.empty()) s->add_reads_filenames(reads);
      s->set_pileup_height(height);
      *s->mutable_variant_caller_options() = vc_opts;
      // Per-sample VC options keep the same thresholds; sample_name in
      // the proto is set by upstream via make_vc_options(sample_name=…)
      // — we bake that here so multi_sample::VariantCaller can identify
      // the target sample from its own VC opts.
      s->mutable_variant_caller_options()->set_sample_name(name);
      for (int o : order) s->add_order(o);
      s->set_skip_output_generation(skip_output);
      if (!small_path.empty()) s->set_small_model_path(small_path);
      if (ds > 0.0) s->set_downsample_fraction(static_cast<float>(ds));
    };

    const bool skip_parents = absl::GetFlag(FLAGS_skip_parent_calling);

    // Order in `samples_in_order`: [parent1, child, parent2].
    // Each sample's `order` controls the channel-stack permutation when
    // building its OWN pileup image (so the target sample sits at slot 1
    // in its own image; parent2 swaps the two parents).
    add_sample("parent1", p1_name.empty() ? "parent1" : p1_name,
               parent1_reads, parent_h, ds_parents,
               {0, 1, 2}, skip_parents,
               absl::GetFlag(FLAGS_small_model_path_parent));
    add_sample("child", sample_name,
               absl::GetFlag(FLAGS_reads), child_h, ds_child,
               {0, 1, 2}, /*skip_output=*/false,
               absl::GetFlag(FLAGS_small_model_path_child));
    add_sample("parent2", p2_name.empty() ? "parent2" : p2_name,
               parent2_reads, parent_h, ds_parents,
               {2, 1, 0}, skip_parents,
               absl::GetFlag(FLAGS_small_model_path_parent));

    // MAIN_SAMPLE_INDEX = 1 (child) per deeptrio/make_examples.py:48.
    opts.set_main_sample_index(1);
    opts.set_sample_role_to_train("child");
  } else {
    SampleOptions* sopt = opts.add_sample_options();
    sopt->set_role("sample");
    sopt->set_name(sample_name);
    sopt->add_reads_filenames(absl::GetFlag(FLAGS_reads));
    sopt->set_pileup_height(100);  // WGS default pileup height per sample.
    *sopt->mutable_variant_caller_options() = vc_opts;
    sopt->mutable_variant_caller_options()->set_sample_name(sample_name);
    opts.set_main_sample_index(0);
    opts.set_sample_role_to_train("sample");
  }

  opts.set_variant_caller(MakeExamplesOptions::VERY_SENSITIVE_CALLER);
  opts.set_realigner_enabled(false);
  opts.set_phase_reads(false);
  opts.set_stream_examples(false);

  return opts;
}

// Returns true when --reads_parent1 was set (trio mode active).
bool IsTrioMode() {
  return !absl::GetFlag(FLAGS_reads_parent1).empty();
}

// Infer sample name from the first RG:SM field in the BAM header.
std::string InferSampleName(
    const nucleus::genomics::v1::SamHeader& header) {
  for (const auto& rg : header.read_groups()) {
    if (!rg.sample_id().empty()) return rg.sample_id();
  }
  return "sample";
}

// Walk a 51-bp window of AlleleCounts around the candidate and populate
// the candidate's allele_frequency_at_position map with VAF (×100, integer)
// at each position. The map is used by the small_model's VAF-context
// features (offsets −25..+25 around the variant).
void PopulateVafContext(
    DeepVariantCall* candidate,
    const std::vector<AlleleCount>& allele_counts) {
  if (allele_counts.empty()) return;
  const int64_t variant_pos = candidate->variant().start();
  const int64_t region_start = allele_counts.front().position().position();
  const int64_t local_idx = variant_pos - region_start;
  constexpr int kHalfWindow = kSmallModelVafContextWindow / 2;  // 25
  for (int o = -kHalfWindow; o <= kHalfWindow; ++o) {
    const int64_t idx = local_idx + o;
    if (idx < 0 || idx >= static_cast<int64_t>(allele_counts.size())) continue;
    const auto& ac = allele_counts[idx];
    const int depth = ac.ref_supporting_read_count() + ac.read_alleles_size();
    const int vaf = depth > 0 ? (100 * ac.read_alleles_size()) / depth : 0;
    (*candidate->mutable_allele_frequency_at_position())[
        ac.position().position()] = vaf;
  }
}

// Returns true if the (alt_idx-only) sub-variant is a SNP — used to pick
// the small_model GQ threshold (snp=20 vs indel=28).
bool IsSnpAlt(const nucleus::genomics::v1::Variant& v, int alt_idx) {
  if (alt_idx < 0 || alt_idx >= v.alternate_bases_size()) return false;
  return v.reference_bases().size() == 1 &&
         v.alternate_bases(alt_idx).size() == 1;
}

// Multi-index version: SNP iff REF is 1 base AND every alt in
// `alt_indices` is 1 base. Mirror of nucleus/util/variant_utils.is_snp(
// variant, exclude_alleles) where exclude_alleles is the complement of
// alt_indices.
bool IsSnpForIndices(const nucleus::genomics::v1::Variant& v,
                      const std::vector<int>& alt_indices) {
  if (alt_indices.empty()) return false;
  if (v.reference_bases().size() != 1) return false;
  for (int idx : alt_indices) {
    if (idx < 0 || idx >= v.alternate_bases_size()) return false;
    if (v.alternate_bases(idx).size() != 1) return false;
  }
  return true;
}

// Phred = -10 * log10(p), truncated toward zero. Capped at 99.
//
// Truncation (not std::round) matches upstream's small_model
// passes_confidence_threshold(ptrue_to_bounded_phred(max_p) >= threshold)
// at the boundary: a phred of 19.5 should *fail* a threshold of 20 (which
// floor-rounds it down to 19), but std::round would push 19.5 up to 20
// and pass — flipping a candidate from big-model dispatch to a
// small_model emit.
int ProbToPhred(double p) {
  if (p <= 0.0) return 99;
  if (p >= 1.0) return 0;
  return std::min(static_cast<int>(-10.0 * std::log10(p)), 99);
}

// Build a CallVariantsOutput proto for a single (candidate, alt_idx) pair
// that the small model has resolved. We tag MID="small_model" in the
// VariantCall.info so postprocess can propagate it to the VCF.
// `alt_indices` may be a single index (single-alt CVO) or two indices
// (multi-alt combo CVO, mirrors upstream's get_set_of_allele_indices
// `multiallelic = combinations(range(N), 2)`).
CallVariantsOutput MakeSmallModelCvo(
    const DeepVariantCall& candidate, const std::vector<int>& alt_indices,
    const float* probs) {
  CallVariantsOutput cvo;
  *cvo.mutable_variant() = candidate.variant();
  for (int idx : alt_indices) cvo.mutable_alt_allele_indices()->add_indices(idx);
  // Probabilities written as double — same wire-format as the big model.
  for (int i = 0; i < 3; ++i) cvo.add_genotype_probabilities(probs[i]);
  // Tag MID in VariantCall.info["MID"]. variant_calling.cc already adds an
  // empty VariantCall, so reuse that slot rather than appending another one
  // (would trigger the VcfWriter's "calls != samples" check).
  auto* v = cvo.mutable_variant();
  if (v->calls_size() == 0) v->add_calls();
  nucleus::SetInfoField("MID", std::string("small_model"),
                         v->mutable_calls(0));
  return cvo;
}

}  // namespace

// Per-thread accumulators returned to the main thread for summing.
struct WorkerStats {
  int64_t total_candidates = 0;
  int64_t total_examples = 0;
  int64_t total_small_hits = 0;
  int64_t total_big_dispatched = 0;
};

int RunMakeExamples(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string reads_path = absl::GetFlag(FLAGS_reads);
  const std::string ref_path = absl::GetFlag(FLAGS_ref);
  const std::string examples_path = absl::GetFlag(FLAGS_examples);

  if (reads_path.empty() || ref_path.empty()) {
    LOG(ERROR) << "Required: --reads, --ref";
    return 1;
  }
  // Trio mode: at least one of --examples / --examples_child must be set.
  // Parents are optional (skip_parent_calling), but child is mandatory.
  if (IsTrioMode()) {
    const std::string ex_child = absl::GetFlag(FLAGS_examples_child);
    if (examples_path.empty() && ex_child.empty()) {
      LOG(ERROR)
          << "Trio mode requires --examples_child (or --examples as alias).";
      return 1;
    }
  } else if (examples_path.empty()) {
    LOG(ERROR) << "Required: --examples";
    return 1;
  }

  const int task_id = absl::GetFlag(FLAGS_task_id);
  const int num_shards = std::max(1, absl::GetFlag(FLAGS_num_shards));
  const int n_threads = std::max(1, absl::GetFlag(FLAGS_threads));

  // ── Open shared reference (only used for header / contigs / sample name
  //     inference). Per-thread workers reopen their own IndexedFastaReader
  //     so AlleleCounter calls into htslib stay thread-local. ──────────────
  auto ref_or = nucleus::IndexedFastaReader::FromFile(
      ref_path, absl::StrCat(ref_path, ".fai"));
  CHECK(ref_or.ok()) << "Failed to open reference: " << ref_path;
  auto ref_reader_main = std::move(ref_or.ValueOrDie());

  // ── Infer sample name from the BAM header (cheap, single read). ──────────
  nucleus::genomics::v1::SamReaderOptions sam_opts;
  sam_opts.mutable_read_requirements()->set_min_mapping_quality(
      absl::GetFlag(FLAGS_min_mapping_quality));
  {
    auto sam_or = nucleus::SamReader::FromFile(reads_path, sam_opts);
    CHECK(sam_or.ok()) << "Failed to open BAM: " << reads_path;
    auto tmp_reader = std::move(sam_or.ValueOrDie());
    std::string sn = absl::GetFlag(FLAGS_sample_name);
    if (sn.empty()) {
      sn = InferSampleName(tmp_reader->Header());
      LOG(INFO) << "Inferred sample name: " << sn;
      absl::SetFlag(&FLAGS_sample_name, sn);
    }
  }
  const std::string sample_name = absl::GetFlag(FLAGS_sample_name);

  // ── Build MakeExamplesOptions ─────────────────────────────────────────────
  const MakeExamplesOptions opts = BuildOptions(sample_name, task_id, num_shards);

  // ── Build calling regions ─────────────────────────────────────────────────
  const auto& contigs = ref_reader_main->Contigs();
  std::vector<std::string> inc_regions, exc_regions;
  {
    const std::string regions_str = absl::GetFlag(FLAGS_regions);
    if (!regions_str.empty()) {
      inc_regions = absl::StrSplit(regions_str, absl::ByAnyChar(" \t,"),
                                   absl::SkipEmpty());
    }
    const std::string excl_str = absl::GetFlag(FLAGS_exclude_regions);
    if (!excl_str.empty()) {
      exc_regions = absl::StrSplit(excl_str, absl::ByAnyChar(" \t,"),
                                   absl::SkipEmpty());
    }
  }
  auto all_regions = BuildCallingRegions(contigs, inc_regions, exc_regions);
  // Partition into chunks of partition_size bp (default 1000), then shard.
  // Mirrors upstream's `regions.partition()` step. Required for realigner
  // window-set parity: each chunk runs the WindowSelector + DBG
  // independently, and adjacent chunks emit overlapping windows at the
  // chunk boundary — without partitioning we'd merge windows across chunk
  // boundaries that upstream keeps separate.
  const int64_t partition_size_bp =
      static_cast<int64_t>(absl::GetFlag(FLAGS_partition_size));
  auto partitioned = PartitionRegions(all_regions, partition_size_bp);
  auto shard_regions = ShardRegions(partitioned, task_id, num_shards);

  LOG(INFO) << "Processing " << shard_regions.size() << " regions (shard "
            << task_id << "/" << num_shards << ", threads=" << n_threads
            << ")";

  const std::string small_path = absl::GetFlag(FLAGS_small_model);
  const std::string small_cvo_path =
      absl::GetFlag(FLAGS_small_model_cvo_outfile);
  const int snp_gq_threshold = absl::GetFlag(FLAGS_small_model_snp_gq_threshold);
  const int indel_gq_threshold =
      absl::GetFlag(FLAGS_small_model_indel_gq_threshold);
  if (!small_path.empty() && small_cvo_path.empty()) {
    LOG(ERROR) << "--small_model requires --small_model_cvo_outfile";
    return 1;
  }

  // ── Atomic region cursor: workers fetch_add to claim regions. ────────────
  std::atomic<size_t> next_region{0};

  // Per-thread output paths. We use the standard `name-NNNNN-of-NNNNN`
  // shard naming so downstream stages that already understand the `@N`
  // shard spec (call_variants / postprocess via TFRecordReader) can read
  // the per-thread files directly — no end-of-stage concat needed.
  //
  // examples_path: if it already carries an `@N` suffix, we honour the
  // caller's N; otherwise we synthesise `examples_path@n_threads` and
  // shard from it. n_threads==1 collapses to the plain path.
  std::string examples_spec = examples_path;
  std::string small_cvo_spec = small_cvo_path;
  if (n_threads > 1) {
    if (examples_spec.find('@') == std::string::npos) {
      examples_spec = absl::StrCat(examples_path, "@", n_threads);
    }
    if (!small_cvo_spec.empty() &&
        small_cvo_spec.find('@') == std::string::npos) {
      small_cvo_spec = absl::StrCat(small_cvo_path, "@", n_threads);
    }
  }
  auto thread_examples_path = [&](int t) {
    return n_threads == 1 ? examples_path : ShardName(examples_spec, t);
  };
  auto thread_small_cvo_path = [&](int t) {
    return n_threads == 1 ? small_cvo_path : ShardName(small_cvo_spec, t);
  };

  // ──────────────────────────────────────────────────────────────────
  // Trio worker — mirrors deeptrio/make_examples.py's per-region loop.
  // Opens 3 SamReaders, builds 3 AlleleCounters per region, runs
  // multi_sample::VariantCaller once per target sample, generates
  // examples with the target's `order` permutation, and writes per-
  // sample small_cvo + examples streams. The single-sample path
  // below is unchanged (preserves the WGS chr20 100% FILTER parity
  // gate already achieved at 5.5d/10).
  // ──────────────────────────────────────────────────────────────────
  auto run_trio_worker = [&](int tid, WorkerStats* out_stats) {
    auto t_ref_or = nucleus::IndexedFastaReader::FromFile(
        ref_path, absl::StrCat(ref_path, ".fai"));
    CHECK(t_ref_or.ok()) << "thread " << tid << ": ref reopen failed";
    auto ref_reader = std::move(t_ref_or.ValueOrDie());

    // Per-role context. Index 0=parent1, 1=child, 2=parent2 (mirrors
    // upstream samples_in_order at deeptrio/make_examples.py:318).
    struct SampleCtx {
      std::string role;
      std::string name;
      std::vector<int> order;          // pileup channel-stack permutation
      int pileup_height = 100;
      bool skip_output = false;
      std::unique_ptr<nucleus::SamReader> sam_reader;
      std::unique_ptr<SmallModel> small_model;
      std::unique_ptr<TFRecordWriter> small_cvo_writer;
      std::string examples_path;
      // Per-target call_variants_outputs counters reported back as stats.
      int64_t total_candidates = 0;
      int64_t total_examples = 0;
      int64_t total_small_hits = 0;
      int64_t total_big_dispatched = 0;
    };
    std::array<SampleCtx, 3> ctx;
    for (int s = 0; s < 3; ++s) {
      const auto& so = opts.sample_options(s);
      ctx[s].role = so.role();
      ctx[s].name = so.name();
      ctx[s].pileup_height = so.pileup_height();
      ctx[s].skip_output = so.skip_output_generation();
      for (int o : so.order()) ctx[s].order.push_back(o);
      // Open BAM (only if reads_filenames is set; pangenome's pangenome-
      // sample has no BAM, but trio always has all 3).
      if (so.reads_filenames_size() > 0) {
        auto sr_or = nucleus::SamReader::FromFile(so.reads_filenames(0),
                                                    sam_opts);
        CHECK(sr_or.ok()) << "thread " << tid << " " << ctx[s].role
                           << ": BAM reopen failed: " << so.reads_filenames(0);
        ctx[s].sam_reader = std::move(sr_or.ValueOrDie());
      }
    }

    // Per-target small_model + small_cvo writer + examples path.
    // Trio convention: child uses --small_model_path_child; parent1/2
    // share --small_model_path_parent. Per-sample examples paths come
    // from --examples_{child,parent1,parent2} (or fall back to
    // --examples for child for backward compat).
    auto trio_examples_path = [&](const std::string& role) -> std::string {
      std::string base;
      if (role == "child")
        base = absl::GetFlag(FLAGS_examples_child).empty()
                   ? examples_path
                   : absl::GetFlag(FLAGS_examples_child);
      else if (role == "parent1")
        base = absl::GetFlag(FLAGS_examples_parent1);
      else  // parent2
        base = absl::GetFlag(FLAGS_examples_parent2);
      if (base.empty()) return "";
      return n_threads == 1 ? base : ShardName(base, tid);
    };
    auto trio_small_cvo_path = [&](const std::string& role) -> std::string {
      std::string base;
      if (role == "child")
        base = absl::GetFlag(FLAGS_small_model_cvo_outfile_child);
      else if (role == "parent1")
        base = absl::GetFlag(FLAGS_small_model_cvo_outfile_parent1);
      else
        base = absl::GetFlag(FLAGS_small_model_cvo_outfile_parent2);
      if (base.empty()) return "";
      return n_threads == 1 ? base : ShardName(base, tid);
    };

    const std::string sm_child  = absl::GetFlag(FLAGS_small_model_path_child);
    const std::string sm_parent = absl::GetFlag(FLAGS_small_model_path_parent);

    std::unordered_map<std::string, std::string> example_filenames;
    for (auto& c : ctx) {
      c.examples_path = trio_examples_path(c.role);
      if (!c.skip_output && !c.examples_path.empty()) {
        example_filenames[c.role] = c.examples_path;
      }
      // Small-model load — child or parent share-of-two depending on role.
      const std::string& sm_path = (c.role == "child") ? sm_child : sm_parent;
      if (!sm_path.empty() && !c.skip_output) {
        c.small_model = SmallModel::Load(sm_path);
        CHECK(c.small_model) << "thread " << tid << " " << c.role
                              << ": small_model load failed: " << sm_path;
        const std::string scp = trio_small_cvo_path(c.role);
        if (!scp.empty()) {
          c.small_cvo_writer = TFRecordWriter::New(scp);
          CHECK(c.small_cvo_writer)
              << "thread " << tid << " " << c.role
              << ": small CVO writer open failed: " << scp;
        }
      }
    }

    multi_sample::VariantCaller caller(
        opts.sample_options(opts.main_sample_index()).variant_caller_options());

    ExamplesGenerator generator(opts, example_filenames);

    while (true) {
      const size_t i = next_region.fetch_add(1, std::memory_order_relaxed);
      if (i >= shard_regions.size()) break;
      const auto& region = shard_regions[i];
      LOG(INFO) << "Trio region: " << region.reference_name() << ":"
                << region.start() << "-" << region.end();

      // Per-sample: query reads, reservoir-sample, store working_reads
      // alongside each ctx. (Realigner is intentionally NOT yet wired
      // for trio — upstream's joint_realignment / per_sample_realignment
      // is a Step-1.3-bis follow-up. For chr20 quick-start fixtures most
      // candidates come from straightforward pileups so realigner-off
      // gives us the first stage-by-stage diff baseline. WGS path keeps
      // realigner enabled.)
      std::array<std::vector<nucleus::genomics::v1::Read>, 3> reads_per_sample_v;
      const int max_rpp = static_cast<int>(opts.max_reads_per_partition());
      for (int s = 0; s < 3; ++s) {
        if (!ctx[s].sam_reader) continue;
        auto reads_or = ctx[s].sam_reader->Query(region);
        if (!reads_or.ok()) {
          LOG(WARNING) << "Query failed for " << ctx[s].role << " "
                       << region.reference_name() << ":" << region.start()
                       << "-" << region.end() << " — " << reads_or.status();
          continue;
        }
        auto& reads_iter = reads_or.ValueOrDie();
        std::vector<nucleus::genomics::v1::Read>& reads =
            reads_per_sample_v[s];
        nucleus::genomics::v1::Read tmp_read;
        while (true) {
          auto next = reads_iter->Next(&tmp_read);
          if (!next.ok() || !next.ValueOrDie()) break;
          reads.push_back(tmp_read);
        }
        reads_iter->Release().IgnoreError();
        if (max_rpp > 0 && reads.size() > static_cast<size_t>(max_rpp)) {
          ::deepvariant::npr::NumpyMt19937 region_rng(opts.random_seed());
          auto sampled = ::deepvariant::npr::ReservoirSamplePtrs(
              reads, max_rpp, region_rng);
          std::vector<nucleus::genomics::v1::Read> kept;
          kept.reserve(sampled.size());
          for (const auto* p : sampled) kept.push_back(*p);
          reads = std::move(kept);
        }
      }

      // Build 3 AlleleCounters keyed by sample_name. Two-pass: probe
      // (no candidate positions) → collect candidate positions → re-
      // build with candidate_positions for ref-read tracking. Same
      // as single-sample path; here we two-pass per sample.
      std::array<std::unique_ptr<AlleleCounter>, 3> counters;
      // First pass: probe per sample.
      std::vector<int> all_candidate_positions;
      for (int s = 0; s < 3; ++s) {
        if (!ctx[s].sam_reader) continue;
        AlleleCounter probe(ref_reader.get(), region, /*positions=*/{},
                            opts.allele_counter_options());
        for (const auto& r : reads_per_sample_v[s]) {
          probe.Add(r, ctx[s].name);
        }
        // Collect probe candidate positions across all samples — the
        // multi_sample::VariantCaller will join across samples; we
        // give every counter the union of candidate positions so
        // ref-read tracking works for joint candidates from another
        // sample's evidence.
        // (Mirror of upstream make_examples_core.py:_make_allele_counter_for_region
        // which uses the joint candidate set for all samples.)
        // Rough approximation: get probe candidates positions; we'll
        // union after the loop.
        // Simpler: rebuild every counter with same merged positions.
      }
      // Simpler approach (chr20 quick-start fixture): probe → union of
      // candidate positions → rebuild each AlleleCounter with the
      // unioned positions. Matches upstream's "positions known up-front"
      // pattern.
      for (int s = 0; s < 3; ++s) {
        if (!ctx[s].sam_reader) continue;
        AlleleCounter probe(ref_reader.get(), region, /*positions=*/{},
                            opts.allele_counter_options());
        for (const auto& r : reads_per_sample_v[s]) {
          probe.Add(r, ctx[s].name);
        }
        // No public position-extractor on AlleleCounter; iterate Counts().
        for (const auto& ac : probe.Counts()) {
          if (!ac.read_alleles().empty()) {
            all_candidate_positions.push_back(
                static_cast<int>(ac.position().position()));
          }
        }
      }
      std::sort(all_candidate_positions.begin(),
                all_candidate_positions.end());
      all_candidate_positions.erase(
          std::unique(all_candidate_positions.begin(),
                      all_candidate_positions.end()),
          all_candidate_positions.end());

      for (int s = 0; s < 3; ++s) {
        if (!ctx[s].sam_reader) continue;
        counters[s] = std::make_unique<AlleleCounter>(
            ref_reader.get(), region, all_candidate_positions,
            opts.allele_counter_options());
        for (const auto& r : reads_per_sample_v[s]) {
          counters[s]->Add(r, ctx[s].name);
        }
      }

      // Build the unordered_map<sample_name, AlleleCounter*> map for
      // multi_sample::VariantCaller.
      std::unordered_map<std::string, AlleleCounter*> ac_map;
      for (int s = 0; s < 3; ++s) {
        if (counters[s]) ac_map[ctx[s].name] = counters[s].get();
      }

      // For each target sample (child, parent1, parent2): generate
      // candidates with the multi-sample API, run small_model dispatch,
      // emit examples + CVOs. Skip parents when --skip_parent_calling.
      for (int s = 0; s < 3; ++s) {
        SampleCtx& C = ctx[s];
        if (C.skip_output) continue;

        std::vector<DeepVariantCall> candidates =
            caller.CallsFromAlleleCounts(ac_map, C.name, C.role);
        if (candidates.empty()) continue;
        C.total_candidates += candidates.size();

        // VAF context — uses the target sample's AlleleCounts.
        if (C.small_model && counters[s]) {
          const auto& allele_counts = counters[s]->Counts();
          for (auto& c : candidates) PopulateVafContext(&c, allele_counts);
        }

        // Small-model dispatch (per alt-set), same as single-sample path.
        std::vector<DeepVariantCall> big_candidates;
        if (C.small_model) {
          for (auto& c : candidates) {
            const int n_alts = c.variant().alternate_bases_size();
            std::vector<std::vector<int>> alt_idx_sets;
            for (int i = 0; i < n_alts; ++i) alt_idx_sets.push_back({i});
            for (int i = 0; i < n_alts; ++i)
              for (int j = i + 1; j < n_alts; ++j)
                alt_idx_sets.push_back({i, j});

            bool any_failed = false;
            c.clear_make_examples_alt_allele_indices();
            for (const auto& idx_set : alt_idx_sets) {
              const auto features = EncodeSmallModelFeatures(c, idx_set);
              float probs[3] = {0, 0, 0};
              bool pred_ok =
                  C.small_model->Predict(features.data(), 1, probs);
              bool accept = false;
              if (pred_ok) {
                const float max_p =
                    std::max({probs[0], probs[1], probs[2]});
                const int gq = ProbToPhred(1.0 - max_p);
                const int threshold = IsSnpForIndices(c.variant(), idx_set)
                                        ? snp_gq_threshold
                                        : indel_gq_threshold;
                accept = (gq >= threshold);
              }
              if (accept) {
                CallVariantsOutput cvo =
                    MakeSmallModelCvo(c, idx_set, probs);
                std::string serialized;
                cvo.SerializeToString(&serialized);
                if (C.small_cvo_writer) {
                  C.small_cvo_writer->WriteRecord(serialized);
                }
                ++C.total_small_hits;
              } else {
                auto* aai = c.add_make_examples_alt_allele_indices();
                for (int idx : idx_set) aai->add_indices(idx);
                any_failed = true;
              }
            }
            if (any_failed) {
              big_candidates.push_back(c);
              ++C.total_big_dispatched;
            }
          }
        } else {
          big_candidates = candidates;
          C.total_big_dispatched += candidates.size();
        }

        if (big_candidates.empty()) continue;

        // ExamplesGenerator: 3 sample read vectors in upstream order
        // [parent1, child, parent2], rendered with this target's order
        // permutation. C.order tells the generator which slot of the
        // 3-sample array to put in slot 1 (target), 0, 2.
        std::vector<nucleus::ConstProtoPtr<DeepVariantCall>> cand_ptrs;
        cand_ptrs.reserve(big_candidates.size());
        for (auto& c : big_candidates) {
          cand_ptrs.push_back(
              nucleus::ConstProtoPtr<DeepVariantCall>(&c));
        }
        std::array<std::vector<nucleus::ConstProtoPtr<
            nucleus::genomics::v1::Read>>, 3> per_sample_ptrs;
        for (int q = 0; q < 3; ++q) {
          per_sample_ptrs[q].reserve(reads_per_sample_v[q].size());
          for (auto& r : reads_per_sample_v[q]) {
            per_sample_ptrs[q].push_back(
                nucleus::ConstProtoPtr<nucleus::genomics::v1::Read>(&r));
          }
        }
        std::vector<std::vector<nucleus::ConstProtoPtr<
            nucleus::genomics::v1::Read>>> reads_per_sample = {
                per_sample_ptrs[0], per_sample_ptrs[1], per_sample_ptrs[2]};
        std::vector<float> mean_coverage = {0.0f, 0.0f, 0.0f};
        std::vector<int> image_shape;

        auto stats = generator.WriteExamplesInRegion(
            absl::MakeSpan(cand_ptrs), absl::MakeSpan(reads_per_sample),
            absl::MakeSpan(C.order), C.role,
            absl::MakeSpan(mean_coverage), &image_shape);
        auto n_it = stats.find("n_examples");
        if (n_it != stats.end()) C.total_examples += n_it->second;
      }
    }  // end while next_region

    generator.SignalShardFinished();
    for (auto& c : ctx) {
      if (c.small_cvo_writer) c.small_cvo_writer->Close();
    }

    // Aggregate per-sample stats into the worker totals (sum across
    // the 3 samples — the postprocess stage will re-bucket them later).
    int64_t tot_cand = 0, tot_ex = 0, tot_small = 0, tot_big = 0;
    for (auto& c : ctx) {
      tot_cand  += c.total_candidates;
      tot_ex    += c.total_examples;
      tot_small += c.total_small_hits;
      tot_big   += c.total_big_dispatched;
    }
    out_stats->total_candidates    = tot_cand;
    out_stats->total_examples      = tot_ex;
    out_stats->total_small_hits    = tot_small;
    out_stats->total_big_dispatched = tot_big;
  };

  // Worker function: opens its own SamReader/IndexedFastaReader/
  // ExamplesGenerator/SmallModel, then loops fetching regions from
  // `next_region` until the queue is exhausted. Writes only to its own
  // per-thread files; no inter-thread mutation.
  auto run_worker = [&](int tid, WorkerStats* out_stats) {
    if (IsTrioMode()) {
      run_trio_worker(tid, out_stats);
      return;
    }
    auto t_ref_or = nucleus::IndexedFastaReader::FromFile(
        ref_path, absl::StrCat(ref_path, ".fai"));
    CHECK(t_ref_or.ok()) << "thread " << tid << ": ref reopen failed";
    auto ref_reader = std::move(t_ref_or.ValueOrDie());

    auto t_sam_or = nucleus::SamReader::FromFile(reads_path, sam_opts);
    CHECK(t_sam_or.ok()) << "thread " << tid << ": BAM reopen failed";
    auto sam_reader = std::move(t_sam_or.ValueOrDie());

    vcf_candidate_importer::VariantCaller caller(
        opts.sample_options(0).variant_caller_options());

    const std::unordered_map<std::string, std::string> example_filenames = {
        {"sample", thread_examples_path(tid)}};
    ExamplesGenerator generator(opts, example_filenames);

    std::unique_ptr<SmallModel> small_model;
    std::unique_ptr<TFRecordWriter> small_cvo_writer;
    if (!small_path.empty()) {
      small_model = SmallModel::Load(small_path);
      CHECK(small_model) << "thread " << tid << ": small_model load failed";
      small_cvo_writer = TFRecordWriter::New(thread_small_cvo_path(tid));
      CHECK(small_cvo_writer)
          << "thread " << tid << ": small CVO writer open failed";
    }

    int64_t total_candidates = 0;
    int64_t total_examples = 0;
    int64_t total_small_hits = 0;
    int64_t total_big_dispatched = 0;

    while (true) {
      const size_t i = next_region.fetch_add(1, std::memory_order_relaxed);
      if (i >= shard_regions.size()) break;
      const auto& region = shard_regions[i];
    LOG(INFO) << "Region: " << region.reference_name() << ":"
              << region.start() << "-" << region.end();

    // Query reads.
    auto reads_or = sam_reader->Query(region);
    if (!reads_or.ok()) {
      LOG(WARNING) << "Query failed for " << region.reference_name() << ":"
                   << region.start() << "-" << region.end()
                   << " — " << reads_or.status();
      continue;
    }
    auto& reads_iter = reads_or.ValueOrDie();

    // Collect reads into a vector (AlleleCounter needs random access).
    std::vector<nucleus::genomics::v1::Read> reads;
    nucleus::genomics::v1::Read tmp_read;
    while (true) {
      auto next = reads_iter->Next(&tmp_read);
      if (!next.ok() || !next.ValueOrDie()) break;
      reads.push_back(tmp_read);
    }
    reads_iter->Release().IgnoreError();

    // Match upstream make_examples_core.py:partition_reads_etc, which
    // applies Algorithm-R reservoir sampling to cap reads per partition
    // at `max_reads_per_partition` (default 1500). Without this cap,
    // high-coverage regions (chr20:31185000-31186000 has 5686 reads
    // post-filter) blow up the pileup-image evidence and produce a
    // different DP/AD/VAF than Docker → different small_model dispatch
    // → different deepvariant softmax → FILTER drift. The RNG is a
    // NumPy-compatible mt19937 (numpy_mt19937.h) seeded with
    // opts.random_seed (609314161, the upstream default), reset per
    // region — matches `np.random.RandomState(seed)` in
    // make_examples_core.py:2134.
    const int max_rpp = static_cast<int>(opts.max_reads_per_partition());
    if (max_rpp > 0 && reads.size() > static_cast<size_t>(max_rpp)) {
      const size_t orig_n = reads.size();
      ::deepvariant::npr::NumpyMt19937 region_rng(opts.random_seed());
      auto sampled =
          ::deepvariant::npr::ReservoirSamplePtrs(reads, max_rpp, region_rng);
      std::vector<nucleus::genomics::v1::Read> kept;
      kept.reserve(sampled.size());
      for (const auto* p : sampled) kept.push_back(*p);
      reads = std::move(kept);
      LOG(INFO) << "  reservoir-sampled " << orig_n << " → " << reads.size()
                 << " reads (max_reads_per_partition=" << max_rpp << ")";
    }

    LOG(INFO) << "  read " << reads.size() << " reads from BAM";
    if (reads.empty()) continue;

    // ── Optional: realign reads through assembled haplotypes ─────────────
    // Done before any AlleleCounter pass so candidate sweep + ref read
    // tracking see the realigned reads (matches upstream's flow).
    std::vector<nucleus::genomics::v1::Read> working_reads;
    if (absl::GetFlag(FLAGS_realigner_enabled)) {
      // Pre-scan AlleleCounter for the WindowSelector. Upstream
      // (realigner/window_selector.py:_candidates_from_reads) builds
      // a *dedicated* AlleleCounter with WindowSelector-specific
      // requirements (ws_min_mapq=20, ws_min_base_quality=20) over an
      // expanded region (region_expansion_in_bp=20). The candidate-
      // emission AlleleCounter further down uses the looser
      // make_examples thresholds (10/10) — they're separate counters.
      const auto realigner_opts = DefaultRealignerOptions();
      const int expand_bp = realigner_opts.ws_config().region_expansion_in_bp();
      auto contig_or = ref_reader->Contig(region.reference_name());
      const int64_t contig_n =
          contig_or.ok() ? contig_or.ValueOrDie()->n_bases() :
          static_cast<int64_t>(region.end()) + expand_bp;
      nucleus::genomics::v1::Range ws_region;
      ws_region.set_reference_name(region.reference_name());
      ws_region.set_start(std::max<int64_t>(0,
          static_cast<int64_t>(region.start()) - expand_bp));
      ws_region.set_end(std::min<int64_t>(contig_n,
          static_cast<int64_t>(region.end()) + expand_bp));

      AlleleCounterOptions ws_ac_opts;
      ws_ac_opts.set_partition_size(opts.allele_counter_options().partition_size());
      ws_ac_opts.mutable_read_requirements()->set_min_mapping_quality(
          realigner_opts.ws_config().min_mapq());
      ws_ac_opts.mutable_read_requirements()->set_min_base_quality(
          realigner_opts.ws_config().min_base_quality());
      ws_ac_opts.mutable_read_requirements()->set_min_base_quality_mode(
          nucleus::genomics::v1::ReadRequirements::ENFORCED_BY_CLIENT);
      // track_ref_reads stays false — the WindowSelector doesn't use ref
      // reads (AlleleFilter() rejects REFERENCE alleles).
      AlleleCounter pre(ref_reader.get(), ws_region, /*candidates=*/{},
                        ws_ac_opts);
      for (const auto& r : reads) pre.Add(r, sample_name);
      working_reads =
          RealignReadsForRegion(reads, ws_region, pre, *ref_reader,
                                 realigner_opts);
      // Optional: dump (qname, contig, pos, mapq, cigar, seq) of
      // post-realigner reads per chunk so we can side-by-side diff
      // against upstream's --emit_realigned_reads BAM.
      static const char* dump_dir = std::getenv("DV_REALIGNED_READS_TSV");
      if (dump_dir) {
        std::string fname = std::string(dump_dir) + "/" +
            region.reference_name() + ":" +
            std::to_string(region.start()) + "-" +
            std::to_string(region.end()) + ".tsv";
        std::ofstream rf(fname);
        if (rf) {
          for (const auto& r : working_reads) {
            rf << r.fragment_name() << '/' << r.read_number() << '\t'
               << r.alignment().position().reference_name() << '\t'
               << r.alignment().position().position() << '\t'
               << r.alignment().mapping_quality() << '\t';
            for (const auto& cu : r.alignment().cigar()) {
              rf << cu.operation_length();
              switch (cu.operation()) {
                using ::nucleus::genomics::v1::CigarUnit;
                case CigarUnit::ALIGNMENT_MATCH:    rf << 'M'; break;
                case CigarUnit::INSERT:             rf << 'I'; break;
                case CigarUnit::DELETE:             rf << 'D'; break;
                case CigarUnit::SKIP:               rf << 'N'; break;
                case CigarUnit::CLIP_SOFT:          rf << 'S'; break;
                case CigarUnit::CLIP_HARD:          rf << 'H'; break;
                case CigarUnit::PAD:                rf << 'P'; break;
                case CigarUnit::SEQUENCE_MATCH:     rf << '='; break;
                case CigarUnit::SEQUENCE_MISMATCH:  rf << 'X'; break;
                default: rf << '?';
              }
            }
            rf << '\t' << r.aligned_sequence() << '\n';
          }
        }
      }
    } else {
      working_reads = reads;
    }

    // First pass: find candidate positions (no ref-read tracking yet).
    AlleleCounter probe(ref_reader.get(), region, {},
                        opts.allele_counter_options());
    for (const auto& r : working_reads) probe.Add(r, sample_name);
    auto probe_candidates = caller.CallsFromAlleleCounter(probe);
    if (probe_candidates.empty()) continue;

    // Second pass: rerun AlleleCounter with the candidate positions known
    // up-front. AlleleCounter only retains REF-supporting reads in its
    // read_alleles map at positions that appear in this list (when
    // track_ref_reads=true). Without this two-pass shape the small_model
    // sees num_reads_supports_ref = 0 on every candidate.
    std::vector<int> candidate_positions;
    candidate_positions.reserve(probe_candidates.size());
    for (const auto& c : probe_candidates) {
      candidate_positions.push_back(static_cast<int>(c.variant().start()));
    }
    std::sort(candidate_positions.begin(), candidate_positions.end());
    candidate_positions.erase(
        std::unique(candidate_positions.begin(), candidate_positions.end()),
        candidate_positions.end());

    AlleleCounter counter(ref_reader.get(), region, candidate_positions,
                          opts.allele_counter_options());
    for (const auto& r : working_reads) counter.Add(r, sample_name);

    std::vector<DeepVariantCall> candidates =
        caller.CallsFromAlleleCounter(counter);
    if (candidates.empty()) continue;

    total_candidates += candidates.size();

    // Small-model first-pass dispatch. Mirror of upstream
    // `SmallModelVariantCaller.call_variants` + `make_small_model_examples.
    // get_set_of_allele_indices`:
    //   - For each candidate, enumerate the FULL set of alt-allele-indices:
    //       biallelic   = [(0,), (1,), …, (N-1,)]
    //       multiallelic = list(combinations(range(N), 2))
    //   - Run small_model on each (candidate, alt_indices) PAIR.
    //   - PER-PAIR pass/fail: if pass → emit small_model CVO; if fail →
    //     append to candidate.make_examples_alt_allele_indices so big_model
    //     generates an example for that specific alt-set only. Multiple
    //     pairs from the same candidate can split between small/big.
    std::vector<DeepVariantCall> big_candidates;
    if (small_model) {
      // Populate VAF context for every candidate (the small model's 51
      // VAF-context features need it; the big model doesn't, but it's cheap
      // and keeps both paths producing the same DeepVariantCall shape).
      const auto& allele_counts = counter.Counts();
      for (auto& c : candidates) {
        PopulateVafContext(&c, allele_counts);
      }

      for (auto& c : candidates) {
        const int n_alts = c.variant().alternate_bases_size();
        // Build the list of alt-index sets to query: single + combinations.
        std::vector<std::vector<int>> alt_idx_sets;
        for (int i = 0; i < n_alts; ++i) alt_idx_sets.push_back({i});
        for (int i = 0; i < n_alts; ++i) {
          for (int j = i + 1; j < n_alts; ++j) {
            alt_idx_sets.push_back({i, j});
          }
        }

        // Per alt-index-set: predict + decide.
        bool any_failed = false;
        c.clear_make_examples_alt_allele_indices();
        for (const auto& idx_set : alt_idx_sets) {
          const auto features = EncodeSmallModelFeatures(c, idx_set);
          float probs[3] = {0, 0, 0};
          bool pred_ok = small_model->Predict(features.data(), 1, probs);
          bool accept = false;
          if (pred_ok) {
            const float max_p = std::max({probs[0], probs[1], probs[2]});
            const int gq = ProbToPhred(1.0 - max_p);
            const int threshold = IsSnpForIndices(c.variant(), idx_set)
                                    ? snp_gq_threshold
                                    : indel_gq_threshold;
            accept = (gq >= threshold);
          }
          if (accept) {
            // Single-alt CVOs use idx_set[0]; the multi-alt (i, j) set is
            // emitted with both indices so postprocess merge can route it
            // correctly.
            CallVariantsOutput cvo = MakeSmallModelCvo(c, idx_set, probs);
            std::string serialized;
            cvo.SerializeToString(&serialized);
            small_cvo_writer->WriteRecord(serialized);
            ++total_small_hits;
          } else {
            // Failed → big model generates an example for this exact
            // alt-index-set only.
            auto* aai = c.add_make_examples_alt_allele_indices();
            for (int idx : idx_set) aai->add_indices(idx);
            any_failed = true;
          }
        }
        if (any_failed) {
          big_candidates.push_back(c);
          ++total_big_dispatched;
        }
      }
    } else {
      big_candidates = candidates;
      total_big_dispatched += candidates.size();
    }

    if (big_candidates.empty()) continue;

    // Wrap in ConstProtoPtr for ExamplesGenerator API.
    std::vector<nucleus::ConstProtoPtr<DeepVariantCall>> cand_ptrs;
    cand_ptrs.reserve(big_candidates.size());
    for (auto& c : big_candidates) {
      cand_ptrs.push_back(nucleus::ConstProtoPtr<DeepVariantCall>(&c));
    }

    std::vector<nucleus::ConstProtoPtr<nucleus::genomics::v1::Read>> read_ptrs;
    read_ptrs.reserve(working_reads.size());
    for (auto& r : working_reads) {
      read_ptrs.push_back(
          nucleus::ConstProtoPtr<nucleus::genomics::v1::Read>(&r));
    }
    std::vector<std::vector<
        nucleus::ConstProtoPtr<nucleus::genomics::v1::Read>>>
        reads_per_sample = {read_ptrs};

    std::vector<int> sample_order = {0};
    std::vector<float> mean_coverage = {0.0f};
    std::vector<int> image_shape;

    LOG(INFO) << "  candidates=" << candidates.size()
              << " reads=" << reads.size();

    auto stats = generator.WriteExamplesInRegion(
        absl::MakeSpan(cand_ptrs), absl::MakeSpan(reads_per_sample),
        absl::MakeSpan(sample_order), "sample",
        absl::MakeSpan(mean_coverage), &image_shape);

    auto n_it = stats.find("n_examples");
    if (n_it != stats.end()) total_examples += n_it->second;
    }  // end while next_region

    generator.SignalShardFinished();
    if (small_cvo_writer) small_cvo_writer->Close();

    out_stats->total_candidates = total_candidates;
    out_stats->total_examples = total_examples;
    out_stats->total_small_hits = total_small_hits;
    out_stats->total_big_dispatched = total_big_dispatched;
  };  // end run_worker lambda

  // ── Dispatch workers ─────────────────────────────────────────────────────
  std::vector<WorkerStats> stats_per_thread(n_threads);
  if (n_threads == 1) {
    run_worker(0, &stats_per_thread[0]);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
      workers.emplace_back([&, t] { run_worker(t, &stats_per_thread[t]); });
    }
    for (auto& w : workers) w.join();
  }

  // ── Sum per-thread stats ─────────────────────────────────────────────────
  WorkerStats agg;
  for (const auto& s : stats_per_thread) {
    agg.total_candidates    += s.total_candidates;
    agg.total_examples      += s.total_examples;
    agg.total_small_hits    += s.total_small_hits;
    agg.total_big_dispatched += s.total_big_dispatched;
  }

  // No end-of-stage concat: workers wrote sharded `name-NNNNN-of-NNNNN`
  // files that downstream stages read directly via TFRecordReader's
  // `@N` shard spec expansion.

  LOG(INFO) << "make_examples done: " << agg.total_candidates << " candidates, "
            << agg.total_examples << " examples written"
            << " (small_model_hits=" << agg.total_small_hits
            << ", big_model_dispatched=" << agg.total_big_dispatched
            << ", threads=" << n_threads << ").";
  return 0;
}

}  // namespace deepvariant
