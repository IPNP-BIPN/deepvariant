// Implementation of the 70-feature extractor for the small_model.
// Mirrors deepvariant/small_model/make_small_model_examples.py:FeatureEncoder.

#include "deepvariant/native/small_model_features.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "deepvariant/protos/deepvariant.pb.h"

namespace deepvariant {

using learning::genomics::deepvariant::DeepVariantCall;
using learning::genomics::deepvariant::DeepVariantCall_ReadSupport;
using nucleus::genomics::v1::Variant;

namespace {

// Pull the reads supporting the chosen alt allele indices into a single
// flat vector of ReadSupport pointers.
std::vector<const DeepVariantCall_ReadSupport*> GetAltReadInfos(
    const DeepVariantCall& candidate,
    const std::vector<int>& alt_allele_indices) {
  std::vector<const DeepVariantCall_ReadSupport*> out;
  for (int idx : alt_allele_indices) {
    if (idx < 0 || idx >= candidate.variant().alternate_bases_size()) continue;
    const auto& alt_bases = candidate.variant().alternate_bases(idx);
    auto it = candidate.allele_support_ext().find(alt_bases);
    if (it == candidate.allele_support_ext().end()) continue;
    for (const auto& r : it->second.read_infos()) {
      out.push_back(&r);
    }
  }
  return out;
}

int MeanInt(const std::vector<const DeepVariantCall_ReadSupport*>& reads,
            int (*getter)(const DeepVariantCall_ReadSupport&)) {
  if (reads.empty()) return 0;
  int64_t sum = 0;
  for (const auto* r : reads) sum += getter(*r);
  return static_cast<int>(sum / static_cast<int64_t>(reads.size()));
}

int GetMQ(const DeepVariantCall_ReadSupport& r) { return r.mapping_quality(); }
int GetBQ(const DeepVariantCall_ReadSupport& r) {
  return r.average_base_quality();
}
int GetReverseStrand100(const DeepVariantCall_ReadSupport& r) {
  return r.is_reverse_strand() ? 100 : 0;
}

// SNP detection (roughly variant_utils.is_snp): every alt is a single base
// and ref is a single base.
bool IsSnp(const Variant& v, const std::set<std::string>& exclude) {
  if (v.reference_bases().size() != 1) return false;
  bool any_alt = false;
  for (const auto& a : v.alternate_bases()) {
    if (exclude.count(a)) continue;
    if (a.size() != 1) return false;
    any_alt = true;
  }
  return any_alt;
}

bool IsInsertion(const Variant& v, const std::set<std::string>& exclude) {
  bool any = false;
  for (const auto& a : v.alternate_bases()) {
    if (exclude.count(a)) continue;
    if (a.size() <= v.reference_bases().size()) return false;
    any = true;
  }
  return any;
}

bool IsDeletion(const Variant& v, const std::set<std::string>& exclude) {
  bool any = false;
  for (const auto& a : v.alternate_bases()) {
    if (exclude.count(a)) continue;
    if (a.size() >= v.reference_bases().size()) return false;
    any = true;
  }
  return any;
}

}  // namespace

std::vector<float> EncodeSmallModelFeatures(
    const DeepVariantCall& candidate,
    const std::vector<int>& alt_allele_indices) {
  std::vector<float> features;
  features.reserve(kSmallModelNumFeatures);

  // Excluded alternates: those NOT in alt_allele_indices.
  std::set<std::string> exclude;
  std::set<int> indices_set(alt_allele_indices.begin(),
                             alt_allele_indices.end());
  for (int i = 0; i < candidate.variant().alternate_bases_size(); ++i) {
    if (!indices_set.count(i)) {
      exclude.insert(candidate.variant().alternate_bases(i));
    }
  }

  // Read sets.
  std::vector<const DeepVariantCall_ReadSupport*> ref_reads;
  for (const auto& r : candidate.ref_support_ext().read_infos()) {
    ref_reads.push_back(&r);
  }
  auto alt_reads = GetAltReadInfos(candidate, alt_allele_indices);

  // Compute total depth across all alleles (ref + every allele's alt reads).
  int total_depth = static_cast<int>(ref_reads.size());
  for (const auto& [_, support] : candidate.allele_support_ext()) {
    total_depth += support.read_infos_size();
  }

  // ── BaseFeatures (12) ─────────────────────────────────────────────────────
  const int n_ref = static_cast<int>(ref_reads.size());
  const int n_alt = static_cast<int>(alt_reads.size());
  const int alt_indices_depth = n_ref + n_alt;
  features.push_back(n_ref);                   // num_reads_supports_ref
  features.push_back(n_alt);                   // num_reads_supports_alt
  features.push_back(alt_indices_depth);       // alt_indices_depth
  features.push_back(total_depth);             // total_depth
  // variant_allele_frequency = 100 * n_alt / total_depth
  features.push_back(total_depth > 0 ? (100 * n_alt / total_depth) : 0);
  // alt_indices_variant_allele_frequency = 100 * n_alt / alt_indices_depth
  features.push_back(alt_indices_depth > 0
                          ? (100 * n_alt / alt_indices_depth)
                          : 0);
  features.push_back(MeanInt(ref_reads, GetMQ));   // ref_mapping_quality
  features.push_back(MeanInt(alt_reads, GetMQ));   // alt_mapping_quality
  features.push_back(MeanInt(ref_reads, GetBQ));   // ref_base_quality
  features.push_back(MeanInt(alt_reads, GetBQ));   // alt_base_quality
  features.push_back(MeanInt(ref_reads, GetReverseStrand100));  // ref_rev_strand
  features.push_back(MeanInt(alt_reads, GetReverseStrand100));  // alt_rev_strand

  // ── VariantFeatures (7) ───────────────────────────────────────────────────
  const auto& v = candidate.variant();
  features.push_back(IsSnp(v, exclude) ? 1 : 0);                 // is_snp
  features.push_back(IsInsertion(v, exclude) ? 1 : 0);           // is_insertion
  features.push_back(IsDeletion(v, exclude) ? 1 : 0);            // is_deletion
  // insertion_length: max(0, max over indices of (alt_len - ref_len))
  int ins_len = 0;
  for (int idx : alt_allele_indices) {
    if (idx < 0 || idx >= v.alternate_bases_size()) continue;
    int d = static_cast<int>(v.alternate_bases(idx).size()) -
            static_cast<int>(v.reference_bases().size());
    ins_len = std::max(ins_len, d);
  }
  features.push_back(std::max(0, ins_len));                      // insertion_length
  // deletion_length: max(0, max over indices of (ref_len - alt_len))
  int del_len = 0;
  for (int idx : alt_allele_indices) {
    if (idx < 0 || idx >= v.alternate_bases_size()) continue;
    int d = static_cast<int>(v.reference_bases().size()) -
            static_cast<int>(v.alternate_bases(idx).size());
    del_len = std::max(del_len, d);
  }
  features.push_back(std::max(0, del_len));                      // deletion_length
  features.push_back(v.alternate_bases_size() > 1 ? 1 : 0);      // is_multiallelic
  features.push_back(alt_allele_indices.size() > 1 ? 1 : 0);     // is_multiple_alt_alleles

  // ── VAF context (51 features, offsets -25..+25 inclusive) ─────────────────
  const auto& vaf_at_pos = candidate.allele_frequency_at_position();
  const int half = kSmallModelVafContextWindow / 2;  // 25
  for (int o = -half; o <= half; ++o) {
    const int64_t pos = v.start() + o;
    auto it = vaf_at_pos.find(static_cast<int>(pos));
    features.push_back(it != vaf_at_pos.end() ? it->second : 0);
  }

  return features;
}

}  // namespace deepvariant
