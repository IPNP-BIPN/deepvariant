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
#include "deepvariant/native/small_model_features.h"
#include "deepvariant/native/small_model_inference.h"
#include "deepvariant/native/tfrecord.h"
#include "deepvariant/protos/deepvariant.pb.h"
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
ABSL_FLAG(int, min_mapping_quality, 10, "Min read mapping quality.");
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

// Phred = -10 * log10(p).  Capped at 99.
int ProbToPhred(double p) {
  if (p <= 0.0) return 99;
  if (p >= 1.0) return 0;
  return std::min(static_cast<int>(std::round(-10.0 * std::log10(p))), 99);
}

// Build a CallVariantsOutput proto for a single (candidate, alt_idx) pair
// that the small model has resolved. We tag MID="small_model" in the
// VariantCall.info so postprocess can propagate it to the VCF.
CallVariantsOutput MakeSmallModelCvo(
    const DeepVariantCall& candidate, int alt_idx,
    const float* probs) {
  CallVariantsOutput cvo;
  *cvo.mutable_variant() = candidate.variant();
  cvo.mutable_alt_allele_indices()->add_indices(alt_idx);
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
  // Single-sample variant_calling.cc — the multi-sample variant emits ~4×
  // more candidates than upstream's pipeline does at the same options,
  // which we don't want. variant_calling.cc has been patched to populate
  // mapping_quality / average_base_quality / is_reverse_strand on the
  // ReadSupport entries (mirror of what multisample does) so the
  // small_model's per-read features aren't saturated at 0.
  // PopulateVafContext below fills in allele_frequency_at_position which
  // the small_model uses for its 51 VAF-context features.
  vcf_candidate_importer::VariantCaller caller(
      opts.sample_options(0).variant_caller_options());

  // ── Optional: small model first-pass ──────────────────────────────────────
  std::unique_ptr<SmallModel> small_model;
  std::unique_ptr<TFRecordWriter> small_cvo_writer;
  const std::string small_path = absl::GetFlag(FLAGS_small_model);
  const std::string small_cvo_path =
      absl::GetFlag(FLAGS_small_model_cvo_outfile);
  const int snp_gq_threshold = absl::GetFlag(FLAGS_small_model_snp_gq_threshold);
  const int indel_gq_threshold =
      absl::GetFlag(FLAGS_small_model_indel_gq_threshold);
  if (!small_path.empty()) {
    if (small_cvo_path.empty()) {
      LOG(ERROR) << "--small_model requires --small_model_cvo_outfile";
      return 1;
    }
    small_model = SmallModel::Load(small_path);
    if (!small_model) {
      LOG(ERROR) << "Failed to load small_model at " << small_path;
      return 1;
    }
    small_cvo_writer = TFRecordWriter::New(small_cvo_path);
    if (!small_cvo_writer) {
      LOG(ERROR) << "Failed to open small CVO writer: " << small_cvo_path;
      return 1;
    }
    LOG(INFO) << "Small model active: " << small_path;
  }

  // ── Main loop ─────────────────────────────────────────────────────────────
  int64_t total_candidates = 0;
  int64_t total_examples = 0;
  int64_t total_small_hits = 0;
  int64_t total_big_dispatched = 0;

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

    // Optional small-model first-pass dispatch. We try each candidate against
    // the small model (one prediction per alt allele); if EVERY alt for a
    // candidate gets a confident enough genotype call, we emit the per-alt
    // CVOs directly and skip the big model. Otherwise the candidate falls
    // through to ExamplesGenerator (which generates the pileup image and the
    // big model picks up downstream in call_variants).
    std::vector<DeepVariantCall> big_candidates;
    if (small_model) {
      // Populate VAF context for every candidate (the small model's 51
      // VAF-context features need it; the big model doesn't, but it's cheap
      // and keeps both paths producing the same DeepVariantCall shape).
      const auto& allele_counts = counter.Counts();
      for (auto& c : candidates) {
        PopulateVafContext(&c, allele_counts);
      }

      for (const auto& c : candidates) {
        const int n_alts = c.variant().alternate_bases_size();
        // Decide per-alt; only keep the candidate fully on the small-model
        // path if every alt clears its threshold.
        std::vector<CallVariantsOutput> per_alt_cvos;
        bool all_alts_pass = (n_alts > 0);
        for (int alt_idx = 0; alt_idx < n_alts; ++alt_idx) {
          const auto features = EncodeSmallModelFeatures(c, {alt_idx});
          float probs[3] = {0, 0, 0};
          if (!small_model->Predict(features.data(), 1, probs)) {
            all_alts_pass = false;
            break;
          }
          // The small_model GQ is the phred score of NOT being the called
          // genotype, i.e. -10·log10(1 − max_p).
          const float max_p = std::max({probs[0], probs[1], probs[2]});
          const int gq = ProbToPhred(1.0 - max_p);
          const int threshold =
              IsSnpAlt(c.variant(), alt_idx) ? snp_gq_threshold
                                              : indel_gq_threshold;
          if (gq < threshold) {
            all_alts_pass = false;
            break;
          }
          per_alt_cvos.push_back(MakeSmallModelCvo(c, alt_idx, probs));
        }
        if (all_alts_pass) {
          for (const auto& cvo : per_alt_cvos) {
            std::string serialized;
            cvo.SerializeToString(&serialized);
            small_cvo_writer->WriteRecord(serialized);
            ++total_small_hits;
          }
        } else {
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
  if (small_cvo_writer) small_cvo_writer->Close();

  LOG(INFO) << "make_examples done: " << total_candidates << " candidates, "
            << total_examples << " examples written"
            << " (small_model_hits=" << total_small_hits
            << ", big_model_dispatched=" << total_big_dispatched << ").";
  return 0;
}

}  // namespace deepvariant
