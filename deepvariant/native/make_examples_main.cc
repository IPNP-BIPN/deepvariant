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
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "deepvariant/allelecounter.h"
#include "deepvariant/make_examples_native.h"
#include "deepvariant/native/regions.h"
#include "deepvariant/protos/deepvariant.pb.h"
#include "deepvariant/variant_calling.h"
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
#include "third_party/nucleus/util/proto_ptr.h"

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
ABSL_FLAG(int, min_mapping_quality, 10, "Min read mapping quality.");
ABSL_FLAG(int, min_base_quality, 10, "Min base quality.");

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

  // Sample options.
  SampleOptions* sopt = opts.add_sample_options();
  sopt->set_role("sample");
  sopt->set_name(sample_name);
  sopt->add_reads_filenames(absl::GetFlag(FLAGS_reads));
  sopt->set_pileup_height(100);  // WGS default pileup height per sample.
  *sopt->mutable_variant_caller_options() = vc_opts;
  opts.set_main_sample_index(0);
  opts.set_sample_role_to_train("sample");

  opts.set_variant_caller(MakeExamplesOptions::VERY_SENSITIVE_CALLER);
  opts.set_realigner_enabled(false);
  opts.set_phase_reads(false);
  opts.set_stream_examples(false);

  return opts;
}

// Infer sample name from the first RG:SM field in the BAM header.
std::string InferSampleName(
    const nucleus::genomics::v1::SamHeader& header) {
  for (const auto& rg : header.read_groups()) {
    if (!rg.sample_id().empty()) return rg.sample_id();
  }
  return "sample";
}

}  // namespace

int RunMakeExamples(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string reads_path = absl::GetFlag(FLAGS_reads);
  const std::string ref_path = absl::GetFlag(FLAGS_ref);
  const std::string examples_path = absl::GetFlag(FLAGS_examples);

  if (reads_path.empty() || ref_path.empty() || examples_path.empty()) {
    LOG(ERROR) << "Required: --reads, --ref, --examples";
    return 1;
  }

  const int task_id = absl::GetFlag(FLAGS_task_id);
  const int num_shards = std::max(1, absl::GetFlag(FLAGS_num_shards));

  // ── Open reference (for AlleleCounter) ────────────────────────────────────
  auto ref_or = nucleus::IndexedFastaReader::FromFile(
      ref_path, absl::StrCat(ref_path, ".fai"));
  CHECK(ref_or.ok()) << "Failed to open reference: " << ref_path;
  auto ref_reader = std::move(ref_or.ValueOrDie());

  // ── Infer sample name ─────────────────────────────────────────────────────
  nucleus::genomics::v1::SamReaderOptions sam_opts;
  sam_opts.mutable_read_requirements()->set_min_mapping_quality(
      absl::GetFlag(FLAGS_min_mapping_quality));
  auto sam_or = nucleus::SamReader::FromFile(reads_path, sam_opts);
  CHECK(sam_or.ok()) << "Failed to open BAM: " << reads_path;
  auto sam_reader = std::move(sam_or.ValueOrDie());

  std::string sample_name = absl::GetFlag(FLAGS_sample_name);
  if (sample_name.empty()) {
    sample_name = InferSampleName(sam_reader->Header());
    LOG(INFO) << "Inferred sample name: " << sample_name;
  }

  // ── Build MakeExamplesOptions ─────────────────────────────────────────────
  const MakeExamplesOptions opts = BuildOptions(sample_name, task_id, num_shards);

  // ── Build calling regions ─────────────────────────────────────────────────
  const auto& contigs = ref_reader->Contigs();
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
  auto shard_regions = ShardRegions(all_regions, task_id, num_shards);

  LOG(INFO) << "Processing " << shard_regions.size() << " regions (shard "
            << task_id << "/" << num_shards << ")";

  // ── ExamplesGenerator ─────────────────────────────────────────────────────
  const std::unordered_map<std::string, std::string> example_filenames = {
      {"sample", examples_path}};
  ExamplesGenerator generator(opts, example_filenames);

  // ── Variant caller ────────────────────────────────────────────────────────
  vcf_candidate_importer::VariantCaller caller(
      opts.sample_options(0).variant_caller_options());

  // ── Main loop ─────────────────────────────────────────────────────────────
  int64_t total_candidates = 0;
  int64_t total_examples = 0;

  for (const auto& region : shard_regions) {
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

    LOG(INFO) << "  read " << reads.size() << " reads from BAM";
    if (reads.empty()) continue;

    // AlleleCounter.
    AlleleCounter counter(ref_reader.get(), region, {},
                          opts.allele_counter_options());
    for (const auto& r : reads) {
      counter.Add(r, sample_name);
    }

    // Candidate variants.
    std::vector<DeepVariantCall> candidates =
        caller.CallsFromAlleleCounter(counter);
    if (candidates.empty()) continue;

    total_candidates += candidates.size();

    // Wrap in ConstProtoPtr for ExamplesGenerator API.
    std::vector<nucleus::ConstProtoPtr<DeepVariantCall>> cand_ptrs;
    cand_ptrs.reserve(candidates.size());
    for (auto& c : candidates) {
      cand_ptrs.push_back(nucleus::ConstProtoPtr<DeepVariantCall>(&c));
    }

    std::vector<nucleus::ConstProtoPtr<nucleus::genomics::v1::Read>> read_ptrs;
    read_ptrs.reserve(reads.size());
    for (auto& r : reads) {
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
  }

  generator.SignalShardFinished();

  LOG(INFO) << "make_examples done: " << total_candidates << " candidates, "
            << total_examples << " examples written.";
  return 0;
}

}  // namespace deepvariant
