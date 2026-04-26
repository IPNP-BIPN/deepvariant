#include "deepvariant/native/cli.h"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

// `run` subcommand flags. Reuse flags declared in the subcommand files
// (--reads, --ref, --regions, --batch_size, --num_shards, etc.) to avoid
// duplicate symbols at link time.
ABSL_FLAG(std::string, model_type, "WGS",
          "Model type: WGS, WES, PACBIO, ONT, HYBRID_PACBIO_ILLUMINA");
ABSL_FLAG(std::string, output_vcf, "", "Output VCF path (run mode).");
ABSL_FLAG(std::string, output_gvcf, "",
          "Output gVCF path (run mode, optional).");
ABSL_FLAG(std::string, intermediate_results_dir, "/tmp/dv_run",
          "Directory for intermediate TFRecord files.");
ABSL_FLAG(std::string, model, "",
          "Path to .mlpackage model (overrides --model_type lookup).");
ABSL_FLAG(std::string, small_model_path, "",
          "Path to small_model .mlpackage. Empty = no small-model first-pass; "
          "every candidate goes through the big InceptionV3.");

// Flags owned by the subcommand .cc files — declared here for use by RunAll.
ABSL_DECLARE_FLAG(std::string, reads);
ABSL_DECLARE_FLAG(std::string, ref);
ABSL_DECLARE_FLAG(std::string, regions);
ABSL_DECLARE_FLAG(int, num_shards);
ABSL_DECLARE_FLAG(int, batch_size);

namespace deepvariant {

namespace {

// Build a flag-vector for a subcommand, splicing in the given extra flags.
std::vector<char*> MakeArgv(const std::string& prog,
                             const std::vector<std::string>& extras) {
  static std::vector<std::string> storage;
  storage.clear();
  storage.push_back(prog);
  for (const auto& e : extras) storage.push_back(e);
  std::vector<char*> argv;
  for (auto& s : storage) argv.push_back(const_cast<char*>(s.c_str()));
  argv.push_back(nullptr);
  return argv;
}

std::string ModelPath(const std::string& model_type) {
  if (!absl::GetFlag(FLAGS_model).empty()) {
    return absl::GetFlag(FLAGS_model);
  }
  // Default install path from deepvariant-models Homebrew formula.
  const char* prefix = std::getenv("DEEPVARIANT_MODELS_DIR");
  std::string base = prefix ? prefix : "/opt/homebrew/share/deepvariant-models";
  std::string type = model_type;
  // Normalise to lowercase.
  for (char& c : type) c = static_cast<char>(std::tolower(c));
  return absl::StrCat(base, "/", type, ".mlpackage");
}

}  // namespace

int RunAll(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_type = absl::GetFlag(FLAGS_model_type);
  const std::string reads_flag = absl::GetFlag(FLAGS_reads);
  const std::string ref_flag = absl::GetFlag(FLAGS_ref);
  const std::string output_vcf_flag = absl::GetFlag(FLAGS_output_vcf);
  const std::string regions_flag = absl::GetFlag(FLAGS_regions);
  const std::string tmp_dir = absl::GetFlag(FLAGS_intermediate_results_dir);
  const int num_shards = absl::GetFlag(FLAGS_num_shards);

  if (reads_flag.empty() || ref_flag.empty() || output_vcf_flag.empty()) {
    LOG(ERROR) << "Usage: deepvariant run --reads=<BAM> --ref=<FASTA> "
                  "--output_vcf=<VCF> [--model_type=WGS] [--regions=chr20]";
    return 1;
  }

  // For num_shards == 1 we use a plain path (no @1 suffix); for >1 shards
  // make_examples writes to one file per task_id and the shard expansion
  // happens later by convention "path-NNNNN-of-NNNNN".
  std::string examples_pattern;
  std::string cvo_pattern;
  std::string small_cvo_path;
  std::string merged_cvo_path;
  if (num_shards <= 1) {
    examples_pattern = absl::StrCat(tmp_dir, "/examples.tfrecord");
    cvo_pattern      = absl::StrCat(tmp_dir, "/cvo.tfrecord");
    small_cvo_path   = absl::StrCat(tmp_dir, "/small_cvo.tfrecord");
    merged_cvo_path  = absl::StrCat(tmp_dir, "/merged_cvo.tfrecord");
  } else {
    examples_pattern = absl::StrCat(tmp_dir, "/examples.tfrecord@", num_shards);
    cvo_pattern      = absl::StrCat(tmp_dir, "/cvo.tfrecord@",      num_shards);
    small_cvo_path   = absl::StrCat(tmp_dir, "/small_cvo.tfrecord");
    merged_cvo_path  = absl::StrCat(tmp_dir, "/merged_cvo.tfrecord");
  }
  const std::string model_path = ModelPath(model_type);
  const std::string small_model_path = absl::GetFlag(FLAGS_small_model_path);

  // ── Stage 1: make_examples ────────────────────────────────────────────────
  LOG(INFO) << "Stage 1: make_examples";
  for (int shard = 0; shard < num_shards; ++shard) {
    std::vector<std::string> me_args = {
        absl::StrCat("--reads=", reads_flag),
        absl::StrCat("--ref=", ref_flag),
        absl::StrCat("--examples=", examples_pattern),
        absl::StrCat("--task_id=", shard),
        absl::StrCat("--num_shards=", num_shards),
    };
    if (!regions_flag.empty()) {
      me_args.push_back(absl::StrCat("--regions=", regions_flag));
    }
    if (!small_model_path.empty()) {
      me_args.push_back(absl::StrCat("--small_model=", small_model_path));
      me_args.push_back(absl::StrCat("--small_model_cvo_outfile=",
                                      small_cvo_path));
    }
    auto argv_me = MakeArgv("deepvariant_make_examples", me_args);
    int n = static_cast<int>(argv_me.size()) - 1;
    if (int rc = RunMakeExamples(n, argv_me.data()); rc != 0) {
      LOG(ERROR) << "make_examples failed (shard " << shard << ")";
      return rc;
    }
  }

  // ── Stage 2: call_variants ────────────────────────────────────────────────
  LOG(INFO) << "Stage 2: call_variants";
  {
    std::vector<std::string> cv_args = {
        absl::StrCat("--examples=", examples_pattern),
        absl::StrCat("--outfile=", cvo_pattern),
        absl::StrCat("--checkpoint=", model_path),
        absl::StrCat("--batch_size=", absl::GetFlag(FLAGS_batch_size)),
    };
    auto argv_cv = MakeArgv("deepvariant_call_variants", cv_args);
    int n = static_cast<int>(argv_cv.size()) - 1;
    if (int rc = RunCallVariants(n, argv_cv.data()); rc != 0) {
      LOG(ERROR) << "call_variants failed";
      return rc;
    }
  }

  // ── Stage 2.5: merge small_cvo + big_cvo into a single file ──────────────
  // postprocess takes one --infile so we concatenate the (already valid)
  // TFRecord files. TFRecord allows naive byte concatenation since each
  // record is self-delimiting.
  std::string postprocess_input = cvo_pattern;
  if (!small_model_path.empty()) {
    LOG(INFO) << "Stage 2.5: merge small_cvo + big_cvo → " << merged_cvo_path;
    std::ifstream sm(small_cvo_path, std::ios::binary);
    std::ifstream bg(cvo_pattern,    std::ios::binary);
    std::ofstream out(merged_cvo_path, std::ios::binary);
    if (!out) {
      LOG(ERROR) << "Cannot open merged CVO: " << merged_cvo_path;
      return 1;
    }
    if (sm) out << sm.rdbuf();
    if (bg) out << bg.rdbuf();
    out.close();
    postprocess_input = merged_cvo_path;
  }

  // ── Stage 3: postprocess_variants ────────────────────────────────────────
  LOG(INFO) << "Stage 3: postprocess_variants";
  {
    std::vector<std::string> pp_args = {
        absl::StrCat("--infile=", postprocess_input),
        absl::StrCat("--ref=", ref_flag),
        absl::StrCat("--output_vcf_outfile=", output_vcf_flag),
    };
    const std::string gvcf_flag = absl::GetFlag(FLAGS_output_gvcf);
    if (!gvcf_flag.empty()) {
      pp_args.push_back(absl::StrCat("--gvcf_outfile=", gvcf_flag));
    }
    auto argv_pp = MakeArgv("deepvariant_postprocess", pp_args);
    int n = static_cast<int>(argv_pp.size()) - 1;
    if (int rc = RunPostprocessVariants(n, argv_pp.data()); rc != 0) {
      LOG(ERROR) << "postprocess_variants failed";
      return rc;
    }
  }

  LOG(INFO) << "Done. VCF: " << output_vcf_flag;
  return 0;
}

}  // namespace deepvariant

int main(int argc, char** argv) {
  absl::InitializeLog();
  // Default log level: send INFO to stderr.
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  if (argc < 2) {
    LOG(ERROR) << "Usage: deepvariant <subcommand> [flags]\n"
                  "Subcommands: make_examples  call_variants  "
                  "postprocess_variants  run";
    return 1;
  }

  const std::string sub(argv[1]);
  // Shift argv so subcommand sees its own flags.
  argv[1] = argv[0];
  int new_argc = argc - 1;
  char** new_argv = argv + 1;

  if (sub == "make_examples") {
    return deepvariant::RunMakeExamples(new_argc, new_argv);
  } else if (sub == "call_variants") {
    return deepvariant::RunCallVariants(new_argc, new_argv);
  } else if (sub == "postprocess_variants") {
    return deepvariant::RunPostprocessVariants(new_argc, new_argv);
  } else if (sub == "run") {
    return deepvariant::RunAll(new_argc, new_argv);
  } else {
    LOG(ERROR) << "Unknown subcommand: " << sub;
    return 1;
  }
}
