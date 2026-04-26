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
#include <set>
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
// Default 20.0 matches upstream postprocess_variants.py default. When a
// CNN RefCall has GQ < this, upstream rewrites it to "./.": NoCall (no
// determination, low confidence). We mirror that exactly.
ABSL_FLAG(double, cnn_homref_call_min_gq, 20.0,
          "All CNN RefCalls whose GQ is less than this become ./. NoCall "
          "instead of 0/0 RefCall (matches upstream default 20.0).");

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
// Truncates toward zero (matching upstream's vcf_conversion.cc, which
// converts the double-valued Log10PErrorToPhred() into a std::vector<int>
// via implicit narrowing rather than std::round).
int ProbToPhred(double p) {
  if (p >= 1.0) return 0;
  if (p <= 0.0) return kMaxPhred;
  int phred = static_cast<int>(-10.0 * std::log10(p));
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

// QUAL of an alt allele = -10 * log10(p_ref). Mirror of
// postprocess_variants.py:compute_quals(predictions, 0) → qual.
double AltAlleleQual(const CallVariantsOutput& cvo) {
  if (cvo.genotype_probabilities_size() < 1) return 0.0;
  const double p_ref = cvo.genotype_probabilities(0);
  if (p_ref <= 0.0) return kMaxPhred;
  if (p_ref >= 1.0) return 0.0;
  return std::min(-10.0 * std::log10(p_ref),
                  static_cast<double>(kMaxPhred));
}

// Returns the set of alt-allele strings to remove from the variant.
// Mirror of postprocess_variants.py:get_alt_alleles_to_remove. An alt is
// flagged for removal when its QUAL (= phred(p_ref)) is below qual_filter.
// If every alt would be removed, the one with the highest QUAL is kept.
std::set<std::string> AltsToRemove(
    const std::vector<const CallVariantsOutput*>& cvos,
    double qual_filter) {
  std::set<std::string> to_remove;
  if (qual_filter <= 0.0 || cvos.empty()) return to_remove;
  const auto& canonical = cvos.front()->variant();
  std::string max_qual_allele;
  double max_qual = -1.0;
  for (const auto* cvo : cvos) {
    const auto& indices = cvo->alt_allele_indices().indices();
    if (indices.size() != 1) continue;
    const int idx = indices[0];
    if (idx < 0 || idx >= canonical.alternate_bases_size()) continue;
    const std::string& alt = canonical.alternate_bases(idx);
    const double qual = AltAlleleQual(*cvo);
    if (qual > max_qual) {
      max_qual = qual;
      max_qual_allele = alt;
    }
    if (qual < qual_filter) to_remove.insert(alt);
  }
  if (!max_qual_allele.empty() &&
      static_cast<int>(to_remove.size()) ==
          canonical.alternate_bases_size()) {
    to_remove.erase(max_qual_allele);  // keep the strongest one
  }
  return to_remove;
}

// Combine all CVOs for one site into a per-genotype likelihood vector.
// Mirror of postprocess_variants.py:merge_predictions "product" mode.
//
// For each diploid genotype (allele1, allele2), each CVO contributes
// cvo.probs[overlap] where overlap = #{alleles in cvo's alt set}, computed
// per allele1, allele2 ∈ {ref, alt1, alt2, …}. Per-CVO contributions are
// fused by product, then normalised across all genotypes.
//
// PL ordering (VCF "G" Number): F(j/k) = k*(k+1)/2 + j  (j ≤ k).
std::vector<double> CombineLikelihoods(
    const std::vector<const CallVariantsOutput*>& cvos, int n_alts) {
  const int n_gt = NumDiploidGenotypes(n_alts);
  std::vector<double> like(n_gt, 1.0);  // multiplicative identity

  if (cvos.empty()) return like;
  // All CVOs of a site share the same `variant` (ADD_HET_ALT_IMAGES); take
  // the alt list from the first.
  const auto& alts = cvos.front()->variant().alternate_bases();

  auto pl_idx = [](int j, int k) {
    if (j > k) std::swap(j, k);
    return k * (k + 1) / 2 + j;
  };

  // Genotype 0 = REF, alleles 1..n_alts = alternate_bases[0..n_alts-1].
  // For the "in this CVO's alt set" check we need each cvo's set of alt
  // strings (from alt_allele_indices).
  std::vector<std::set<std::string>> per_cvo_alts;
  per_cvo_alts.reserve(cvos.size());
  for (const auto* cvo : cvos) {
    std::set<std::string> s;
    for (int idx : cvo->alt_allele_indices().indices()) {
      if (idx >= 0 && idx < alts.size()) s.insert(alts[idx]);
    }
    per_cvo_alts.push_back(std::move(s));
  }

  // For every diploid genotype, fuse probabilities across CVOs by product.
  for (int k = 0; k <= n_alts; ++k) {
    for (int j = 0; j <= k; ++j) {
      const std::string a1 = (j == 0) ? "" : alts[j - 1];  // "" = REF
      const std::string a2 = (k == 0) ? "" : alts[k - 1];
      double fused = 1.0;
      for (size_t ci = 0; ci < cvos.size(); ++ci) {
        const auto& probs = cvos[ci]->genotype_probabilities();
        if (probs.size() < 3) continue;
        const int overlap = (a1.empty() ? 0 : per_cvo_alts[ci].count(a1)) +
                            (a2.empty() ? 0 : per_cvo_alts[ci].count(a2));
        // overlap ∈ {0, 1, 2} maps directly to the 3-class softmax index.
        fused *= probs[overlap];
      }
      like[pl_idx(j, k)] = fused;
    }
  }

  // Normalise — but only when product fusion happened across multiple
  // CVOs. Upstream's merge_predictions returns the raw predictions for
  // single-CVO sites (the common case in WGS) and only renormalises after
  // product fusion. For single-CVO sites the FP32 softmax already saturates
  // some predictions to exactly 1.0; renormalising by the full-precision
  // sum (=1.0+ε) sneaks the called probability slightly below 1.0, which
  // pushes ptrue_to_bounded_phred away from the 99-cap and gives
  // off-by-many GQ values.
  if (cvos.size() > 1) {
    double s = 0;
    for (double v : like) s += v;
    if (s <= 0.0) {
      std::fill(like.begin(), like.end(), 1.0 / n_gt);
    } else {
      for (double& v : like) v /= s;
    }
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
      {"NoCall",
       "Site has no call due to low quality (GQ < cnn_homref_call_min_gq)"},
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
  // Mirror upstream: print QUAL to 1 decimal (e.g. 39.4, not 39.3745).
  wr_opts.set_round_qual_values(true);
  auto writer_or = nucleus::VcfWriter::ToFile(outfile, hdr, wr_opts);
  CHECK(writer_or.ok()) << "Failed to open VCF output: " << outfile;
  auto vcf_writer = std::move(writer_or.ValueOrDie());

  const double qual_filter = absl::GetFlag(FLAGS_qual_filter);
  const double homref_min_gq = absl::GetFlag(FLAGS_cnn_homref_call_min_gq);

  int written = 0;
  int refcall = 0;
  int nocall = 0;

  for (const auto& key : ordered_keys) {
    const auto& cvos = groups[key];
    Variant variant = cvos.front()->variant();
    const int orig_n_alts = variant.alternate_bases_size();
    const int orig_n_gt = NumDiploidGenotypes(orig_n_alts);

    // Compute alt-pruning set on the ORIGINAL alts (CVOs still reference
    // them by index). We do the actual pruning AFTER picking the
    // best genotype.
    const auto alts_to_remove = AltsToRemove(cvos, qual_filter);

    // Combine likelihoods over the ORIGINAL alt list.
    auto like = CombineLikelihoods(cvos, orig_n_alts);

    // Mask out genotypes whose alleles are in alts_to_remove. Setting
    // their likelihood to 0 makes them not selectable as argmax.
    if (!alts_to_remove.empty()) {
      for (int k = 0; k <= orig_n_alts; ++k) {
        for (int j = 0; j <= k; ++j) {
          const std::string a1 =
              (j == 0) ? "" : variant.alternate_bases(j - 1);
          const std::string a2 =
              (k == 0) ? "" : variant.alternate_bases(k - 1);
          if ((!a1.empty() && alts_to_remove.count(a1)) ||
              (!a2.empty() && alts_to_remove.count(a2))) {
            like[k * (k + 1) / 2 + j] = 0.0;
          }
        }
      }
      // Renormalise.
      double s = 0;
      for (double v : like) s += v;
      if (s > 0.0) for (double& v : like) v /= s;
    }

    // Now physically prune the variant (renumbering alts). Preserve all
    // other fields — VariantCall.info contains DP/AD/VAF set in
    // make_examples; we must NOT throw them away by replacing the proto.
    if (!alts_to_remove.empty()) {
      // Compute which original alt indices survive — index ranges from 0
      // (first alt) to n_alts-1.
      std::vector<bool> keep_alt(orig_n_alts, false);
      {
        const auto& orig_alts = variant.alternate_bases();
        for (int i = 0; i < orig_alts.size(); ++i) {
          keep_alt[i] = !alts_to_remove.count(orig_alts.Get(i));
        }
      }
      google::protobuf::RepeatedPtrField<std::string> kept_alts;
      for (const auto& a : variant.alternate_bases()) {
        if (!alts_to_remove.count(a)) *kept_alts.Add() = a;
      }
      *variant.mutable_alternate_bases() = std::move(kept_alts);

      // Mirror upstream's AlleleRemapper.reindex_allele_indexed_fields for
      // _ALT_ALLELE_INDEXED_FORMAT_FIELDS = {("AD", true), ("VAF", false),
      // ("MF", true), ("MD", true)}. AD/MF/MD have a ref entry at index 0
      // (ref_is_zero=true) so keep [0] + the kept alt slots. VAF has no ref
      // entry (ref_is_zero=false) so it just gets the kept alt slots.
      for (auto& call : *variant.mutable_calls()) {
        auto* info = call.mutable_info();
        for (const auto& field_info :
             {std::make_pair(std::string("AD"), true),
              std::make_pair(std::string("VAF"), false),
              std::make_pair(std::string("MF"), true),
              std::make_pair(std::string("MD"), true)}) {
          auto it = info->find(field_info.first);
          if (it == info->end()) continue;
          ::nucleus::genomics::v1::ListValue kept;
          const bool ref_is_zero = field_info.second;
          const auto& vals = it->second.values();
          for (int i = 0; i < vals.size(); ++i) {
            bool keep;
            if (ref_is_zero && i == 0) {
              keep = true;  // always keep the ref entry
            } else {
              const int orig_alt = ref_is_zero ? (i - 1) : i;
              keep = (orig_alt < orig_n_alts) ? keep_alt[orig_alt] : false;
            }
            if (keep) *kept.add_values() = vals.Get(i);
          }
          *it->second.mutable_values() = std::move(*kept.mutable_values());
        }
      }
    }

    const int n_alts = variant.alternate_bases_size();
    if (n_alts == 0) continue;
    const int n_gt = NumDiploidGenotypes(n_alts);

    // After pruning, remap the original-index likelihood vector down to
    // the new alt indexing. (Genotype (j, k) on pruned alts maps back to
    // (j', k') on the original alts where j', k' are the original
    // positions of the j-th and k-th non-pruned alts.)
    std::vector<int> new_to_orig(n_alts + 1);
    new_to_orig[0] = 0;
    {
      int new_pos = 1;
      for (int orig = 0; orig < orig_n_alts; ++orig) {
        if (!alts_to_remove.count(variant.alternate_bases().Get(
                std::min(new_pos - 1, n_alts - 1)))) {
          // Find the original index of variant.alternate_bases(new_pos - 1)
          // in the source CVO's alt list.
          // Since `variant` post-prune lists alts in original order, the
          // mapping for new index i is the i-th surviving original index.
        }
      }
      // Simpler reconstruction: walk pruned alts and find each in the
      // first CVO's alt list.
      const auto& orig_alts = cvos.front()->variant().alternate_bases();
      int n = 1;
      for (int i = 0; i < n_alts; ++i) {
        for (int oi = 0; oi < orig_alts.size(); ++oi) {
          if (orig_alts.Get(oi) == variant.alternate_bases(i)) {
            new_to_orig[n++] = oi + 1;
            break;
          }
        }
      }
    }
    std::vector<double> like_pruned(n_gt, 0.0);
    for (int k = 0; k <= n_alts; ++k) {
      for (int j = 0; j <= k; ++j) {
        const int oj = new_to_orig[j];
        const int ok = new_to_orig[k];
        const int new_idx = k * (k + 1) / 2 + j;
        const int orig_idx =
            std::max(oj, ok) * (std::max(oj, ok) + 1) / 2 + std::min(oj, ok);
        if (orig_idx < orig_n_gt) {
          like_pruned[new_idx] = like[orig_idx];
        }
      }
    }
    // Renormalise — but only when alts were actually pruned (the masked
    // genotypes leave the vector summing to <1). For non-pruned single-CVO
    // sites the FP32 saturation in the small_model output already means
    // predictions[0] == 1.0 exactly; renormalising by sum=1.0+ε would push
    // it below 1, which then makes ptrue_to_bounded_phred miss the 99-cap
    // and emit GQ=78 instead of 99 for very-confident homref calls.
    if (!alts_to_remove.empty()) {
      double sp = 0;
      for (double v : like_pruned) sp += v;
      if (sp > 0.0) for (double& v : like_pruned) v /= sp;
    }
    like = std::move(like_pruned);

    // argmax genotype.
    int best = 0;
    for (int i = 1; i < n_gt; ++i) {
      if (like[i] > like[best]) best = i;
    }
    auto [j, k] = GenotypeFromPLIndex(best, n_alts);

    // QUAL = phred-scale of P(non-ref).
    //
    // Upstream's formula:
    //   qual = ptrue_to_bounded_phred(min(sum(predictions[1:]), 1.0))
    //        = phred(1 - sum(predictions[1:]))
    // *not* phred(predictions[0]) — these only agree when the prediction
    // vector sums to exactly 1.0, which it doesn't quite under FP32. Using
    // predictions[0] directly drifts QUAL by up to ~0.1 (e.g. 54.1 vs 54).
    double sum_alt = 0.0;
    for (int i = 1; i < n_gt; ++i) sum_alt += like[i];
    if (sum_alt > 1.0) sum_alt = 1.0;
    const double err_for_qual = std::max(1.0 - sum_alt, 0.0);
    double qual = (err_for_qual >= 1.0) ? 0.0
                                        : std::min(-10.0 * std::log10(err_for_qual),
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

    // GQ — mirror of postprocess_variants.py:compute_quals's
    //   gq = round(ptrue_to_bounded_phred(predictions[prediction_index]))
    // i.e. phred(1 - P_called), bounded. Different from "second-best
    // probability"; matters at the cnn_homref_call_min_gq=20 boundary.
    const double p_called = like[best];
    int gq;
    if (p_called >= 1.0) {
      gq = kMaxPhred;
    } else {
      // Mirror upstream's ptrue_to_bounded_phred: floor at 1.25e-10 (so
      // max phred is -10*log10(1.25e-10) = 99.0309) and round-to-even
      // (np.around) — std::round would split half-integer ties the wrong
      // way (35.5 → 36 instead of 36).
      const double err = std::max(1.0 - p_called, 1.25e-10);
      gq = static_cast<int>(std::nearbyint(-10.0 * std::log10(err)));
      gq = std::min(std::max(gq, 0), kMaxPhred);
    }
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

    // Mirror postprocess_variants.py:uncall_homref_gt_if_lowqual.
    // CNN RefCalls with GQ < cnn_homref_call_min_gq become "./.": NoCall.
    if (variant.filter_size() == 1 && variant.filter(0) == "RefCall" &&
        gq < homref_min_gq) {
      variant.clear_filter();
      variant.add_filter("NoCall");
      call->clear_genotype();
      call->add_genotype(-1);
      call->add_genotype(-1);
      ++nocall;
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
            << " (" << (refcall - nocall) << " RefCall, "
            << nocall << " NoCall, "
            << (written - refcall) << " PASS).";
  return 0;
}

}  // namespace deepvariant

            