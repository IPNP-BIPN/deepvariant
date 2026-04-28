#include "deepvariant/native/cli.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
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
ABSL_DECLARE_FLAG(std::string, inference_backend);
ABSL_DECLARE_FLAG(std::string, checkpoint);

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

  // Single-process pipeline: one make_examples call (using --threads=N for
  // intra-process parallelism, writing sharded `name-NNNNN-of-NNNNN`
  // files), one call_variants call (sharded examples in, single cvo out),
  // one postprocess.
  const int n_threads = std::max(1, num_shards);
  const std::string examples_base =
      absl::StrCat(tmp_dir, "/examples.tfrecord");
  const std::string examples_pattern =
      n_threads > 1 ? absl::StrCat(examples_base, "@", n_threads)
                    : examples_base;
  const std::string cvo_pattern    = absl::StrCat(tmp_dir, "/cvo.tfrecord");
  const std::string small_cvo_base = absl::StrCat(tmp_dir, "/small_cvo.tfrecord");
  const std::string small_cvo_path =
      n_threads > 1 ? absl::StrCat(small_cvo_base, "@", n_threads)
                    : small_cvo_base;
  const std::string merged_cvo_path =
      absl::StrCat(tmp_dir, "/merged_cvo.tfrecord");

  // For --inference_backend=metal, the user passes a `.dvw` weight bundle
  // via --checkpoint; for coreml (Phase 2), the bundle is a `.mlpackage`
  // resolved by --model or the default ModelPath(model_type) lookup.
  const std::string inference_backend =
      absl::GetFlag(FLAGS_inference_backend);
  const std::string user_checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const std::string user_model = absl::GetFlag(FLAGS_model);
  std::string model_path;
  if (!user_checkpoint.empty()) {
    model_path = user_checkpoint;
  } else if (!user_model.empty()) {
    model_path = user_model;
  } else {
    model_path = ModelPath(model_type);
  }
  const std::string small_model_path = absl::GetFlag(FLAGS_small_model_path);

  // ── Stage 1: make_examples (single in-process call, --threads=N) ─────────
  // Internally fans out N worker threads (each with its own
  // SamReader / IndexedFastaReader / ExamplesGenerator / SmallModel) writing
  // sharded `name-NNNNN-of-NNNNN` files. Downstream stages read them
  // directly via TFRecordReader's `@N` shard expansion — no end-of-stage
  // concat. One process = ~N×100 % CPU under `top`, mirroring
  // salmon/samtools.
  LOG(INFO) << "Stage 1: make_examples (--threads=" << n_threads
            << ", in-process)";
  {
    std::vector<std::string> me_args = {
        absl::StrCat("--reads=", reads_flag),
        absl::StrCat("--ref=", ref_flag),
        absl::StrCat("--examples=", examples_pattern),
        absl::StrCat("--threads=", n_threads),
        // task_id=0/num_shards=1 => the single in-process call owns the
        // whole region set; the worker pool partitions it via atomic.
        "--task_id=0",
        "--num_shards=1",
        "--realigner_enabled=true",
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
      LOG(ERROR) << "make_examples failed";
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
        absl::StrCat("--inference_backend=", inference_backend),
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
  // TFRecord files. TFRecord allows naive byte copy since each record is
  // self-delimiting.
  // small_cvo may be a `name@N` shard spec (one file per make_examples
  // worker thread); expand and concat each shard.
  std::string postprocess_input = cvo_pattern;
  if (!small_model_path.empty()) {
    LOG(INFO) << "Stage 2.5: merge small_cvo + big_cvo → " << merged_cvo_path;
    std::ofstream out(merged_cvo_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      LOG(ERROR) << "Cannot open merged CVO: " << merged_cvo_path;
      return 1;
    }
    auto append_path = [&](const std::string& p) {
      std::ifstream in(p, std::ios::binary);
      if (in) out << in.rdbuf();
    };
    // small_cvo: expand "@N" → per-shard files.
    auto at = small_cvo_path.find('@');
    if (at == std::string::npos) {
      append_path(small_cvo_path);
    } else {
      const std::string prefix = small_cvo_path.substr(0, at);
      int nshard = 0;
      if (!absl::SimpleAtoi(small_cvo_path.substr(at + 1), &nshard) ||
          nshard <= 0) {
        LOG(ERROR) << "Bad small_cvo shard spec: " << small_cvo_path;
        return 1;
      }
      for (int i = 0; i < nshard; ++i) {
        append_path(absl::StrCat(prefix, "-",
                                  absl::Dec(i, absl::kZeroPad5),
                                  "-of-", absl::Dec(nshard, absl::kZeroPad5)));
      }
    }
    // big_cvo: single file (call_variants writes once).
    append_path(cvo_pattern);
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
