// Native postprocess_variants — calling mode only.
//
// Reads CallVariantsOutput TFRecords, sorts by genomic coordinate,
// assigns genotypes from 3-class probabilities, and writes VCF.
//
// Genotype mapping:
//   class 0 → 0/0 (hom ref)
//   class 1 → 0/1 (het)
//   class 2 → 1/1 (hom alt)

#include "deepvariant/native/postprocess_main.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "deepvariant/native/tfrecord.h"
#include "deepvariant/protos/deepvariant.pb.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "third_party/nucleus/io/reference.h"
#include "third_party/nucleus/io/vcf_writer.h"
#include "third_party/nucleus/protos/reference.pb.h"
#include "third_party/nucleus/protos/variants.pb.h"
#include "third_party/nucleus/util/utils.h"

ABSL_FLAG(std::string, infile, "", "Input CVO TFRecord path (may be sharded).");
// --ref and --sample_name are owned by make_examples_main.cc.
ABSL_DECLARE_FLAG(std::string, ref);
ABSL_DECLARE_FLAG(std::string, sample_name);
// --outfile here means VCF; renamed to avoid collision with call_variants.
ABSL_FLAG(std::string, output_vcf_outfile, "", "Output VCF path.");
ABSL_FLAG(std::string, gvcf_outfile, "", "gVCF output path (optional).");
ABSL_FLAG(double, qual_filter, 0.0,
          "Filter calls with QUAL below this threshold.");

namespace deepvariant {

using learning::genomics::deepvariant::CallVariantsOutput;

namespace {

// Expand a sharded filespec like "path/to/file@4" into the 4 shard paths.
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

constexpr int kPhred255 = 255;

// GQ from probability: -10 * log10(1 - prob), capped at 255.
int ProbToGQ(double prob) {
  if (prob >= 1.0) return kPhred255;
  double gq = -10.0 * std::log10(1.0 - prob);
  return static_cast<int>(std::min(gq, static_cast<double>(kPhred255)));
}

// QUAL from call probs: -10 * log10(P(ref/ref)).
double ProbsToQual(const google::protobuf::RepeatedField<double>& probs) {
  if (probs.size() < 1) return 0.0;
  double p_hom_ref = probs[0];
  if (p_hom_ref <= 0.0) return 60.0;
  return std::min(-10.0 * std::log10(p_hom_ref), 60.0);
}

// Assign genotype and quality from 3-class softmax probabilities.
// class 0 = hom ref, class 1 = het, class 2 = hom alt.
void AssignGenotypeFromProbs(
    const google::protobuf::RepeatedField<double>& probs,
    nucleus::genomics::v1::VariantCall* call, double* qual_out) {
  *qual_out = 0.0;
  if (probs.size() < 3) {
    call->add_genotype(0);
    call->add_genotype(0);
    return;
  }

  int argmax = 0;
  float max_p = probs[0];
  for (int i = 1; i < probs.size(); ++i) {
    if (probs[i] > max_p) {
      max_p = probs[i];
      argmax = i;
    }
  }

  *qual_out = ProbsToQual(probs);

  switch (argmax) {
    case 0:
      call->add_genotype(0);
      call->add_genotype(0);
      break;
    case 1:
      call->add_genotype(0);
      call->add_genotype(1);
      break;
    default:
      call->add_genotype(1);
      call->add_genotype(1);
      break;
  }

  // GQ via info map (the VCF FORMAT GQ field is stored in VariantCall.info).
  // Skipped for now — not strictly needed for a well-formed VCF.
  (void)ProbToGQ;
}

// Build a VcfHeader from reference contigs.
nucleus::genomics::v1::VcfHeader MakeVcfHeader(
    const std::vector<nucleus::genomics::v1::ContigInfo>& contigs,
    const std::string& sample_name) {
  nucleus::genomics::v1::VcfHeader hdr;
  hdr.set_fileformat("VCFv4.2");

  // FILTER lines.
  auto* pass_filter = hdr.add_filters();
  pass_filter->set_id("PASS");
  pass_filter->set_description("All filters passed");

  // INFO fields.
  {
    auto* f = hdr.add_infos();
    f->set_id("END");
    f->set_number("1");
    f->set_type("Integer");
    f->set_description("End position (for symbolic alleles)");
  }

  // FORMAT fields.
  {
    auto* f = hdr.add_formats();
    f->set_id("GT");
    f->set_number("1");
    f->set_type("String");
    f->set_description("Genotype");
  }
  {
    auto* f = hdr.add_formats();
    f->set_id("GQ");
    f->set_number("1");
    f->set_type("Integer");
    f->set_description("Genotype quality");
  }
  {
    auto* f = hdr.add_formats();
    f->set_id("DP");
    f->set_number("1");
    f->set_type("Integer");
    f->set_description("Read depth");
  }
  {
    auto* f = hdr.add_formats();
    f->set_id("AD");
    f->set_number("R");
    f->set_type("Integer");
    f->set_description("Allelic depths for ref and alt alleles");
  }
  {
    auto* f = hdr.add_formats();
    f->set_id("VAF");
    f->set_number("A");
    f->set_type("Float");
    f->set_description("Variant allele fractions");
  }
  {
    auto* f = hdr.add_formats();
    f->set_id("PL");
    f->set_number("G");
    f->set_type("Integer");
    f->set_description("Phred-scaled genotype likelihoods");
  }

  // Contigs.
  for (const auto& c : contigs) {
    auto* contig = hdr.add_contigs();
    *contig = c;
  }

  // Sample name.
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

  // Contig → position-in-FASTA map for sorting.
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

  // ── Sort by genomic coordinate ────────────────────────────────────────────
  std::stable_sort(
      cvo_list.begin(), cvo_list.end(),
      [&contig_to_pos](const CallVariantsOutput& a,
                       const CallVariantsOutput& b) {
        const int pa = contig_to_pos.count(a.variant().reference_name())
                           ? contig_to_pos.at(a.variant().reference_name())
                           : INT_MAX;
        const int pb = contig_to_pos.count(b.variant().reference_name())
                           ? contig_to_pos.at(b.variant().reference_name())
                           : INT_MAX;
        if (pa != pb) return pa < pb;
        return a.variant().start() < b.variant().start();
      });

  // ── Open VCF writer ───────────────────────────────────────────────────────
  std::string sample_name = absl::GetFlag(FLAGS_sample_name);
  if (sample_name.empty()) sample_name = "SAMPLE";
  nucleus::genomics::v1::VcfHeader hdr =
      MakeVcfHeader(contigs, sample_name);

  nucleus::genomics::v1::VcfWriterOptions wr_opts;
  auto writer_or = nucleus::VcfWriter::ToFile(outfile, hdr, wr_opts);
  CHECK(writer_or.ok()) << "Failed to open VCF output: " << outfile;
  auto vcf_writer = std::move(writer_or.ValueOrDie());

  const double qual_filter = absl::GetFlag(FLAGS_qual_filter);

  // ── Convert and write variants ────────────────────────────────────────────
  int written = 0;
  int filtered = 0;

  for (const auto& cvo : cvo_list) {
    if (!cvo.has_variant()) continue;

    // Work on a copy since we'll modify it.
    nucleus::genomics::v1::Variant variant = cvo.variant();

    // Find the alt allele set that corresponds to this CVO (the model picks
    // one alt at a time; here we always have one call per CVO).
    double qual = 0.0;

    // Add a VariantCall with the genotype from probabilities.
    if (variant.calls_size() == 0) {
      auto* call = variant.add_calls();
      call->set_call_set_name(sample_name);
      AssignGenotypeFromProbs(cvo.genotype_probabilities(), call, &qual);
    } else {
      // Calls may already be present from make_examples. Update the genotype.
      for (auto& call : *variant.mutable_calls()) {
        call.set_call_set_name(sample_name);
        call.clear_genotype();
        AssignGenotypeFromProbs(cvo.genotype_probabilities(), &call, &qual);
        break;  // Only first call for now.
      }
    }

    variant.set_quality(qual);

    // Apply QUAL filter — pass if qual >= threshold or threshold == 0.
    if (qual_filter > 0.0 && qual < qual_filter) {
      variant.add_filter("lowQUAL");
      ++filtered;
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

  LOG(INFO) << "postprocess_variants done: " << written << " variants written"
            << " (" << filtered << " filtered by QUAL).";
  return 0;
}

}  // namespace deepvariant
