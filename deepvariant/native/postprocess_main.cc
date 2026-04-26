// Native postprocess_variants — calling mode.
//
// Reads CallVariantsOutput TFRecords, groups by genomic site (multi-allelic
// merge), assigns the most-likely diploid genotype, and writes VCF with
// FORMAT fields GT:GQ:DP:AD:VAF:PL.
//
// Multi-allelic merge: upstream make_examples emits one example per
// alt-allele combination at multi-allelic sites (multi_allelic_mode =
// ADD_HET_ALT_IMAGES). Each resulting CVO carries:
//   - the same Variant (with the full alt list)
//   - cvo.alt_allele_indices.indices: which alt(s) the example tested
//   - cvo.genotype_probabilities: 3-vector
//     - if indices == [i]:   [P(0/0), P(0/(i+1)), P((i+1)/(i+1))]
//     - if indices == [i,j]: [P(other), P((i+1)/(j+1)), P(other)]
// We collect these into a likelihood table over all diploid genotypes,
// pick argmax, and emit one VCF line per site.

#include "deepvariant/native/postprocess_main.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "deepvariant/native/tfrecord.h"
#include "deepvariant/protos/deepvariant.pb.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "third_party/nucleus/io/reference.h"
#include "third_party/nucleus/io/vcf_writer.h"
#include "third_party/nucleus/protos/reference.pb.h"
#include "third_party/nucleus/protos/struct.pb.h"
#include "third_party/nucleus/protos/variants.pb.h"
#include "third_party/nucleus/util/utils.h"

ABSL_FLAG(std::string, infile, "", "Input CVO TFRecord path (may be sharded).");
ABSL_DECLARE_FLAG(std::string, ref);
ABSL_DECLARE_FLAG(std::string, sample_name);
ABSL_FLAG(std::string, output_vcf_outfile, "", "Output VCF path.");
ABSL_FLAG(std::string, gvcf_outfile, "", "gVCF output path (optional).");
ABSL_FLAG(double, qual_filter, 1.0,
          "Variants with QUAL below this become RefCall instead of PASS.");

namespace deepvariant {

using learning::genomics::deepvariant::CallVariantsOutput;
using nucleus::genomics::v1::Variant;
using nucleus::genomics::v1::VariantCall;

namespace {

constexpr int kMaxPhred = 99;

std::vector<std::string> ExpandShards(const std::string& spec) {
  auto at = spec.find('@');
  if (at == std::string::npos) return {spec};
  const std::string prefix = spec.substr(0, at);
  int n;
  if (!absl::SimpleAtoi(spec.substr(at + 1), &n) || n <= 0) return {spec};
  std::vector<std::string> paths;
  for (int i = 0; i < n; ++i) {
    paths.push_back(absl::StrCat(prefix, "-", absl::Dec(i, absl::kZeroPad5),
                                  "-of-", absl::Dec(n, absl::kZeroPad5)));
  }
  return paths;
}

// Convert probability p (in [0,1]) to a phred score, capped at 99.
int ProbToPhred(double p) {
  if (p >= 1.0) return 0;
  if (p <= 0.0) return kMaxPhred;
  int phred = static_cast<int>(std::round(-10.0 * std::log10(p)));
  return std::min(std::max(phred, 0), kMaxPhred);
}

// Number of diploid genotypes for a variant with `n_alts` alternates:
// 0/0, 0/1, 1/1, 0/2, 1/2, 2/2, ... = (n_alleles)*(n_alleles+1)/2.
int NumDiploidGenotypes(int n_alts) {
  const int n_alleles = n_alts + 1;
  return n_alleles * (n_alleles + 1) / 2;
}

// Return the two-allele genotype (a, b) with a <= b for the given VCF PL
// index. PL ordering: F(j/k) = k*(k+1)/2 + j  (j <= k).
std::pair<int, int> GenotypeFromPLIndex(int pl_index, int n_alts) {
  for (int k = 0; k <= n_alts; ++k) {
    for (int j = 0; j <= k; ++j) {
      const int idx = k * (k + 1) / 2 + j;
      if (idx == pl_index) return {j, k};
    }
  }
  return {0, 0};  // fallback
}

// Combine all CVOs for one site into a per-genotype likelihood vector.
// Returns a vector of length NumDiploidGenotypes(n_alts).
//
// Each CVO contributes its three-class probs to the corresponding diploid
// genotypes:
//   indices == [i]   →   contributes to (0,0), (0,i+1), (i+1,i+1)
//   indices == [i,j] →   contributes to (i+1,j+1) on probs[1]
//
// When multiple CVOs assign a probability to the same genotype, we take the
// MAX (the most confident measurement of that genotype's likelihood).
std::vector<double> CombineLikelihoods(
    const std::vector<const CallVariantsOutput*>& cvos, int n_alts) {
  const int n_gt = NumDiploidGenotypes(n_alts);
  std::vector<double> like(n_gt, 0.0);

  auto pl_idx = [](int j, int k) {
    if (j > k) std::swap(j, k);
    return k * (k + 1) / 2 + j;
  };

  for (const auto* cvo : cvos) {
    const auto& probs = cvo->genotype_probabilities();
    if (probs.size() < 3) continue;
    const auto& indices = cvo->alt_allele_indices().indices();
    if (indices.size() == 1) {
      const int i = indices[0];
      // (0,0)            ← probs[0]
      // (0, i+1)         ← probs[1]
      // (i+1, i+1)       ← probs[2]
      like[pl_idx(0, 0)] = std::max(like[pl_idx(0, 0)], probs[0]);
      like[pl_idx(0, i + 1)] = std::max(like[pl_idx(0, i + 1)], probs[1]);
      like[pl_idx(i + 1, i + 1)] =
          std::max(like[pl_idx(i + 1, i + 1)], probs[2]);
    } else if (indices.size() == 2) {
      const int i = indices[0];
      const int j = indices[1];
      // (i+1, j+1)       ← probs[1] (the het-of-both signal)
      like[pl_idx(i + 1, j + 1)] =
          std::max(like[pl_idx(i + 1, j + 1)], probs[1]);
      // probs[0] also contributes to ref genotype.
      like[pl_idx(0, 0)] = std::max(like[pl_idx(0, 0)], probs[0]);
    }
  }

  // Renormalise so the vector sums to 1 (or close): if no entry was set,
  // default to uniform.
  double s = 0;
  for (double v : like) s += v;
  if (s <= 0.0) {
    std::fill(like.begin(), like.end(), 1.0 / n_gt);
  } else {
    for (double& v : like) v /= s;
  }
  return like;
}

// Build a VcfHeader from reference contigs.
nucleus::genomics::v1::VcfHeader MakeVcfHeader(
    const std::vector<nucleus::genomics::v1::ContigInfo>& contigs,
    const std::string& sample_name) {
  nucleus::genomics::v1::VcfHeader hdr;
  hdr.set_fileformat("VCFv4.2");

  struct Filt { const char* id; const char* desc; };
  static constexpr Filt kFilters[] = {
      {"PASS",    "All filters passed"},
      {"RefCall", "Most likely homozygous reference"},
      {"LowQual", "Confidence in this variant being real is below threshold"},
  };
  for (const auto& fi : kFilters) {
    auto* f = hdr.add_filters();
    f->set_id(fi.id);
    f->set_description(fi.desc);
  }

  // INFO fields.
  {
    auto* f = hdr.add_infos();
    f->set_id("END");
    f->set_number("1");
    f->set_type("Integer");
    f->set_description("End position (for symbolic alleles)");
  }

  // FORMAT fields.
  struct Fmt {
    const char* id;
    const char* num;
    const char* type;
    const char* desc;
  };
  static constexpr Fmt fmts[] = {
      {"GT",  "1", "String",  "Genotype"},
      {"GQ",  "1", "Integer", "Conditional genotype quality"},
      {"DP",  "1", "Integer", "Read depth"},
      {"AD",  "R", "Integer", "Allelic depths for ref and alt alleles"},
      {"VAF", "A", "Float",   "Variant allele fractions"},
      {"MID", "1", "String",  "Model identifier (small_model | deepvariant)"},
      {"PL",  "G", "Integer", "Phred-scaled genotype likelihoods"},
  };
  for (const auto& f : fmts) {
    auto* fi = hdr.add_formats();
    fi->set_id(f.id);
    fi->set_number(f.num);
    fi->set_type(f.type);
    fi->set_description(f.desc);
  }

  // Contigs.
  for (const auto& c : contigs) {
    *hdr.add_contigs() = c;
  }
  hdr.add_sample_names(sample_name);
  return hdr;
}

}  // namespace

int RunPostprocessVariants(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string infile = absl::GetFlag(FLAGS_infile);
  const std::string outfile = absl::GetFlag(FLAGS_output_vcf_outfile);
  const std::string ref_path = absl::GetFlag(FLAGS_ref);

  if (infile.empty() || outfile.empty() || ref_path.empty()) {
    LOG(ERROR) << "Required: --infile, --output_vcf_outfile, --ref";
    return 1;
  }

  // ── Open reference for contig order ───────────────────────────────────────
  auto ref_or = nucleus::IndexedFastaReader::FromFile(
      ref_path, absl::StrCat(ref_path, ".fai"));
  CHECK(ref_or.ok()) << "Failed to open reference: " << ref_path;
  auto ref_reader = std::move(ref_or.ValueOrDie());
  const auto& contigs = ref_reader->Contigs();

  std::map<std::string, int> contig_to_pos;
  for (int i = 0; i < static_cast<int>(contigs.size()); ++i) {
    contig_to_pos[contigs[i].name()] = i;
  }

  // ── Read all CallVariantsOutput protos ────────────────────────────────────
  const std::vector<std::string> shard_paths = ExpandShards(infile);
  std::vector<CallVariantsOutput> cvo_list;
  for (const auto& path : shard_paths) {
    auto reader = TFRecordReader::New(path);
    if (!reader) {
      LOG(WARNING) << "Cannot open shard: " << path;
      continue;
    }
    while (reader->GetNext()) {
      CallVariantsOutput cvo;
      if (!cvo.ParseFromString(reader->record())) {
        LOG(WARNING) << "Failed to parse CVO proto in " << path;
        continue;
      }
      cvo_list.push_back(std::move(cvo));
    }
    reader->Close();
  }
  LOG(INFO) << "Read " << cvo_list.size() << " CallVariantsOutput protos.";

  // ── Group CVOs by site key (chrom, pos, ref, alts) ────────────────────────
  // The variant proto is identical for all CVOs of the same site under
  // ADD_HET_ALT_IMAGES; only the alt_allele_indices differ.
  using SiteKey = std::tuple<std::string, int64_t, std::string, std::string>;
  std::map<SiteKey, std::vector<const CallVariantsOutput*>> groups;
  for (const auto& cvo : cvo_list) {
    if (!cvo.has_variant()) continue;
    const auto& v = cvo.variant();
    SiteKey k{v.reference_name(), v.start(), v.reference_bases(),
              absl::StrJoin(v.alternate_bases(), ",")};
    groups[k].push_back(&cvo);
  }
  LOG(INFO) << "Grouped into " << groups.size() << " unique sites.";

  // ── Sort sites by genomic coordinate ──────────────────────────────────────
  std::vector<SiteKey> ordered_keys;
  ordered_keys.reserve(groups.size());
  for (const auto& [k, _] : groups) ordered_keys.push_back(k);
  std::sort(ordered_keys.begin(), ordered_keys.end(),
            [&contig_to_pos](const SiteKey& a, const SiteKey& b) {
              const int pa = contig_to_pos.count(std::get<0>(a))
                                 ? contig_to_pos.at(std::get<0>(a))
                                 : INT_MAX;
              const int pb = contig_to_pos.count(std::get<0>(b))
                                 ? contig_to_pos.at(std::get<0>(b))
                                 : INT_MAX;
              if (pa != pb) return pa < pb;
              return std::get<1>(a) < std::get<1>(b);
            });

  // ── Open VCF writer ───────────────────────────────────────────────────────
  std::string sample_name = absl::GetFlag(FLAGS_sample_name);
  if (sample_name.empty()) sample_name = "SAMPLE";
  auto hdr = MakeVcfHeader(contigs, sample_name);
  nucleus::genomics::v1::VcfWriterOptions wr_opts;
  // Tell the writer to read PL from VariantCall.info instead of from the
  // (Float-typed) genotype_likelihood field, which lets us write Integer PL.
  wr_opts.set_retrieve_gl_and_pl_from_info_map(true);
  auto writer_or = nucleus::VcfWriter::ToFile(outfile, hdr, wr_opts);
  CHECK(writer_or.ok()) << "Failed to open VCF output: " << outfile;
  auto vcf_writer = std::move(writer_or.ValueOrDie());

  const double qual_filter = absl::GetFlag(FLAGS_qual_filter);

  int written = 0;
  int refcall = 0;

  for (const auto& key : ordered_keys) {
    const auto& cvos = groups[key];
    Variant variant = cvos.front()->variant();
    const int n_alts = variant.alternate_bases_size();
    const int n_gt = NumDiploidGenotypes(n_alts);

    // Combine likelihoods over all CVOs of this site.
    auto like = CombineLikelihoods(cvos, n_alts);

    // argmax genotype.
    int best = 0;
    for (int i = 1; i < n_gt; ++i) {
      if (like[i] > like[best]) best = i;
    }
    auto [j, k] = GenotypeFromPLIndex(best, n_alts);

    // QUAL = phred-scale of P(0/0).
    double p_ref = like[0];
    double qual = (p_ref >= 1.0) ? 0.0
                                 : std::min(-10.0 * std::log10(p_ref),
                                            static_cast<double>(kMaxPhred));

    // Set up the VariantCall.
    if (variant.calls_size() == 0) variant.add_calls();
    auto* call = variant.mutable_calls(0);
    call->set_call_set_name(sample_name);
    call->clear_genotype();
    call->add_genotype(j);
    call->add_genotype(k);

    // Propagate MID from any of the source CVOs. If at least one CVO in
    // this site's group was tagged as a small_model hit, use that;
    // otherwise fall back to deepvariant. (Both tags are set upstream of
    // postprocess: small_model in make_examples_main.cc, deepvariant in
    // call_variants_main.cc.)
    std::string mid;
    for (const auto* cvo : cvos) {
      for (const auto& src_call : cvo->variant().calls()) {
        auto it = src_call.info().find("MID");
        if (it != src_call.info().end() && it->second.values_size() > 0) {
          const std::string& v = it->second.values(0).string_value();
          if (v == "small_model") { mid = v; break; }
          if (mid.empty()) mid = v;
        }
      }
      if (mid == "small_model") break;
    }
    if (!mid.empty()) {
      nucleus::SetInfoField("MID", mid, call);
    }

    // GQ = quality of the called genotype vs the next-best.
    double second_best = 0.0;
    for (int i = 0; i < n_gt; ++i) {
      if (i != best) second_best = std::max(second_best, like[i]);
    }
    int gq = ProbToPhred(second_best);
    nucleus::SetInfoField("GQ", gq, call);

    // PL = phred-scaled likelihoods for each genotype.
    std::vector<int> pl(n_gt);
    int min_pl = kMaxPhred;
    for (int i = 0; i < n_gt; ++i) {
      pl[i] = ProbToPhred(like[i]);
      min_pl = std::min(min_pl, pl[i]);
    }
    for (int& v : pl) v -= min_pl;
    nucleus::SetInfoField("PL", pl, call);

    variant.set_quality(qual);

    // QUAL filter: low-confidence variants become RefCall.
    if (best == 0 || qual < qual_filter) {
      variant.add_filter("RefCall");
      ++refcall;
    } else {
      variant.add_filter("PASS");
    }

    auto status = vcf_writer->Write(variant);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to write variant at "
                   << variant.reference_name() << ":" << variant.start()
                   << " — " << status;
    } else {
      ++written;
    }
  }

  LOG(INFO) << "postprocess_variants done: " << written << " VCF lines"
            << " (" << refcall << " RefCall, "
            << (written - refcall) << " PASS).";
  return 0;
}

}  // namespace deepvariant

            