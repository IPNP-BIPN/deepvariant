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

// DeepTrio (Step 1.5) — trio mode flags. When --reads_parent1 is set,
// run mode dispatches 3× call_variants with the appropriate child/parent
// model and writes 3 separate VCFs.
ABSL_DECLARE_FLAG(std::string, reads_parent1);
ABSL_DECLARE_FLAG(std::string, reads_parent2);
ABSL_DECLARE_FLAG(std::string, sample_name_parent1);
ABSL_DECLARE_FLAG(std::string, sample_name_parent2);
ABSL_DECLARE_FLAG(std::string, examples_child);
ABSL_DECLARE_FLAG(std::string, examples_parent1);
ABSL_DECLARE_FLAG(std::string, examples_parent2);
ABSL_DECLARE_FLAG(std::string, small_model_path_child);
ABSL_DECLARE_FLAG(std::string, small_model_path_parent);
ABSL_DECLARE_FLAG(std::string, small_model_cvo_outfile_child);
ABSL_DECLARE_FLAG(std::string, small_model_cvo_outfile_parent1);
ABSL_DECLARE_FLAG(std::string, small_model_cvo_outfile_parent2);
ABSL_FLAG(std::string, checkpoint_child, "",
          "Trio mode: model checkpoint (.dvw or .mlpackage) for child.");
ABSL_FLAG(std::string, checkpoint_parent, "",
          "Trio mode: model checkpoint shared by parent1 and parent2.");
ABSL_FLAG(std::string, output_vcf_child, "",
          "Trio mode: output VCF for the child sample.");
ABSL_FLAG(std::string, output_vcf_parent1, "",
          "Trio mode: output VCF for parent1.");
ABSL_FLAG(std::string, output_vcf_parent2, "",
          "Trio mode: output VCF for parent2.");
ABSL_FLAG(std::string, output_gvcf_child, "",
          "Trio mode: output gVCF for the child sample.");
ABSL_FLAG(std::string, output_gvcf_parent1, "",
          "Trio mode: output gVCF for parent1.");
ABSL_FLAG(std::string, output_gvcf_parent2, "",
          "Trio mode: output gVCF for parent2.");

// DeepSomatic (Step 2) — somatic mode flags. When --reads_tumor is set,
// run mode dispatches 1× call_variants on the tumor model and emits a
// single tumor VCF. tumor_only mode = no --reads_normal.
ABSL_DECLARE_FLAG(std::string, reads_tumor);
ABSL_DECLARE_FLAG(std::string, reads_normal);
ABSL_DECLARE_FLAG(std::string, sample_name_tumor);
ABSL_DECLARE_FLAG(std::string, sample_name_normal);
ABSL_DECLARE_FLAG(std::string, examples_tumor);
ABSL_DECLARE_FLAG(std::string, examples_normal);
ABSL_DECLARE_FLAG(std::string, small_model_path_somatic);
ABSL_DECLARE_FLAG(std::string, small_model_cvo_outfile_tumor);
ABSL_DECLARE_FLAG(int, pileup_image_height_tumor);
ABSL_DECLARE_FLAG(int, pileup_image_height_normal);

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

// Forward decls.
int RunAllTrio(int argc, char** argv);
int RunAllSomatic(int argc, char** argv);

int RunAll(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // Trio mode: --reads_parent1 set → dispatch the 3-sample pipeline.
  if (!absl::GetFlag(FLAGS_reads_parent1).empty()) {
    return RunAllTrio(argc, argv);
  }

  // Somatic mode: --reads_tumor set → dispatch the 2-sample (tumor+
  // normal) or 1-sample (tumor_only) pipeline. Single tumor VCF output.
  if (!absl::GetFlag(FLAGS_reads_tumor).empty()) {
    return RunAllSomatic(argc, argv);
  }

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

// ──────────────────────────────────────────────────────────────────────
// Trio dispatch: one make_examples (3 sample streams), 3× call_variants
// (child + parent1 + parent2 with the appropriate child/parent model),
// 3× postprocess (one VCF per sample). Mirrors the upstream
// run_deeptrio.py command sequence at deeptrio-quick-start.md.
// ──────────────────────────────────────────────────────────────────────
int RunAllTrio(int argc, char** argv) {
  // ParseCommandLine is already called by RunAll before dispatch here.
  const std::string ref_flag = absl::GetFlag(FLAGS_ref);
  const std::string regions_flag = absl::GetFlag(FLAGS_regions);
  const std::string tmp_dir = absl::GetFlag(FLAGS_intermediate_results_dir);
  const int num_shards = absl::GetFlag(FLAGS_num_shards);
  const int n_threads = std::max(1, num_shards);

  const std::string reads_child   = absl::GetFlag(FLAGS_reads);
  const std::string reads_parent1 = absl::GetFlag(FLAGS_reads_parent1);
  const std::string reads_parent2 = absl::GetFlag(FLAGS_reads_parent2);
  if (reads_child.empty() || reads_parent1.empty() || reads_parent2.empty()) {
    LOG(ERROR) << "Trio: requires --reads (child), --reads_parent1, --reads_parent2";
    return 1;
  }

  const std::string out_child   = absl::GetFlag(FLAGS_output_vcf_child);
  const std::string out_parent1 = absl::GetFlag(FLAGS_output_vcf_parent1);
  const std::string out_parent2 = absl::GetFlag(FLAGS_output_vcf_parent2);
  if (out_child.empty() || out_parent1.empty() || out_parent2.empty()) {
    LOG(ERROR) << "Trio: requires --output_vcf_child, --output_vcf_parent1, "
                  "--output_vcf_parent2";
    return 1;
  }

  // Resolve per-role model checkpoints. Allow either explicit
  // --checkpoint_child / --checkpoint_parent OR fall back to legacy
  // --checkpoint (used for both — useful for smoke tests with one model).
  std::string ckpt_child  = absl::GetFlag(FLAGS_checkpoint_child);
  std::string ckpt_parent = absl::GetFlag(FLAGS_checkpoint_parent);
  if (ckpt_child.empty())  ckpt_child  = absl::GetFlag(FLAGS_checkpoint);
  if (ckpt_parent.empty()) ckpt_parent = absl::GetFlag(FLAGS_checkpoint);
  if (ckpt_child.empty() || ckpt_parent.empty()) {
    LOG(ERROR) << "Trio: requires --checkpoint_child + --checkpoint_parent "
                  "(or --checkpoint as a shared fallback)";
    return 1;
  }
  const std::string sm_child  = absl::GetFlag(FLAGS_small_model_path_child);
  const std::string sm_parent = absl::GetFlag(FLAGS_small_model_path_parent);

  const std::string inference_backend =
      absl::GetFlag(FLAGS_inference_backend);

  // Per-sample intermediate paths.
  struct PerSamplePaths {
    std::string role;
    std::string examples_pattern;
    std::string small_cvo_pattern;
    std::string cvo_path;
    std::string merged_cvo_path;
    std::string sm_path;        // small_model weights dir (or empty)
    std::string ckpt_path;      // big-model checkpoint
    std::string output_vcf;
    std::string output_gvcf;
  };
  std::array<PerSamplePaths, 3> P;
  P[0].role = "child";
  P[1].role = "parent1";
  P[2].role = "parent2";
  for (auto& p : P) {
    p.examples_pattern  = absl::StrCat(tmp_dir, "/examples_", p.role,
                                        ".tfrecord");
    if (n_threads > 1) {
      p.examples_pattern = absl::StrCat(p.examples_pattern, "@", n_threads);
    }
    p.small_cvo_pattern = absl::StrCat(tmp_dir, "/small_cvo_", p.role,
                                        ".tfrecord");
    if (n_threads > 1) {
      p.small_cvo_pattern =
          absl::StrCat(p.small_cvo_pattern, "@", n_threads);
    }
    p.cvo_path        = absl::StrCat(tmp_dir, "/cvo_", p.role, ".tfrecord");
    p.merged_cvo_path =
        absl::StrCat(tmp_dir, "/merged_cvo_", p.role, ".tfrecord");
  }
  P[0].sm_path = sm_child;   P[0].ckpt_path = ckpt_child;
  P[1].sm_path = sm_parent;  P[1].ckpt_path = ckpt_parent;
  P[2].sm_path = sm_parent;  P[2].ckpt_path = ckpt_parent;
  P[0].output_vcf = out_child;   P[0].output_gvcf = absl::GetFlag(FLAGS_output_gvcf_child);
  P[1].output_vcf = out_parent1; P[1].output_gvcf = absl::GetFlag(FLAGS_output_gvcf_parent1);
  P[2].output_vcf = out_parent2; P[2].output_gvcf = absl::GetFlag(FLAGS_output_gvcf_parent2);

  // ── Stage 1: ONE make_examples invocation produces 3 example streams.
  LOG(INFO) << "Trio Stage 1: make_examples (3-sample, --threads=" << n_threads
            << ")";
  {
    std::vector<std::string> me_args = {
        absl::StrCat("--reads=", reads_child),
        absl::StrCat("--reads_parent1=", reads_parent1),
        absl::StrCat("--reads_parent2=", reads_parent2),
        absl::StrCat("--ref=", ref_flag),
        absl::StrCat("--examples_child=",   P[0].examples_pattern),
        absl::StrCat("--examples_parent1=", P[1].examples_pattern),
        absl::StrCat("--examples_parent2=", P[2].examples_pattern),
        absl::StrCat("--threads=", n_threads),
        "--task_id=0",
        "--num_shards=1",
        // Step 1.3-bis: realigner runs per-sample in the trio worker
        // (mirrors upstream's realign_reads_per_sample_multisample).
        // Closes the candidate-count gap with Docker on indel-rich
        // regions where misalignment otherwise inflates AlleleCounter
        // counts with phantom alleles.
        "--realigner_enabled=true",
    };
    if (!regions_flag.empty()) {
      me_args.push_back(absl::StrCat("--regions=", regions_flag));
    }
    if (!absl::GetFlag(FLAGS_sample_name_parent1).empty()) {
      me_args.push_back(absl::StrCat("--sample_name_parent1=",
                                      absl::GetFlag(FLAGS_sample_name_parent1)));
    }
    if (!absl::GetFlag(FLAGS_sample_name_parent2).empty()) {
      me_args.push_back(absl::StrCat("--sample_name_parent2=",
                                      absl::GetFlag(FLAGS_sample_name_parent2)));
    }
    if (!sm_child.empty()) {
      me_args.push_back(absl::StrCat("--small_model_path_child=", sm_child));
      me_args.push_back(absl::StrCat("--small_model_cvo_outfile_child=",
                                      P[0].small_cvo_pattern));
    }
    if (!sm_parent.empty()) {
      me_args.push_back(absl::StrCat("--small_model_path_parent=", sm_parent));
      me_args.push_back(absl::StrCat("--small_model_cvo_outfile_parent1=",
                                      P[1].small_cvo_pattern));
      me_args.push_back(absl::StrCat("--small_model_cvo_outfile_parent2=",
                                      P[2].small_cvo_pattern));
    }
    // DeepTrio WGS default thresholds (upstream scripts/run_deeptrio.py):
    //   --small_model_snp_gq_threshold 15 (vs WGS default 20)
    //   --small_model_indel_gq_threshold 29 (vs WGS default 28)
    me_args.push_back("--small_model_snp_gq_threshold=15");
    me_args.push_back("--small_model_indel_gq_threshold=29");
    auto argv_me = MakeArgv("deepvariant_make_examples", me_args);
    int n = static_cast<int>(argv_me.size()) - 1;
    if (int rc = RunMakeExamples(n, argv_me.data()); rc != 0) {
      LOG(ERROR) << "Trio: make_examples failed";
      return rc;
    }
  }

  // ── Stage 2-3 per sample: call_variants → merge → postprocess.
  for (auto& p : P) {
    LOG(INFO) << "Trio Stage 2 (" << p.role << "): call_variants";
    {
      // Trio WGS pileup is 140×221×7 (child 60 + 2×parent 40).
      // The make_examples worker rendered with these heights; pass
      // through to call_variants so it builds the right Metal input.
      // TODO(step-1-bis): add --model_type WGS/PACBIO/ONT dispatch
      // to pick 140 vs other shapes per upstream's per-mode defaults.
      std::vector<std::string> cv_args = {
          absl::StrCat("--examples=", p.examples_pattern),
          absl::StrCat("--outfile=", p.cvo_path),
          absl::StrCat("--checkpoint=", p.ckpt_path),
          absl::StrCat("--batch_size=", absl::GetFlag(FLAGS_batch_size)),
          absl::StrCat("--inference_backend=", inference_backend),
          "--input_height=140",
          "--input_channels=7",
      };
      auto argv_cv = MakeArgv("deepvariant_call_variants", cv_args);
      int n = static_cast<int>(argv_cv.size()) - 1;
      if (int rc = RunCallVariants(n, argv_cv.data()); rc != 0) {
        LOG(ERROR) << "Trio: call_variants failed for " << p.role;
        return rc;
      }
    }

    // Stage 2.5: merge small_cvo + big cvo (per sample).
    std::string postprocess_input = p.cvo_path;
    if (!p.sm_path.empty()) {
      LOG(INFO) << "Trio Stage 2.5 (" << p.role << "): merge → "
                << p.merged_cvo_path;
      std::ofstream out(p.merged_cvo_path,
                         std::ios::binary | std::ios::trunc);
      if (!out) {
        LOG(ERROR) << "Cannot open merged CVO: " << p.merged_cvo_path;
        return 1;
      }
      auto append_path = [&](const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (in) out << in.rdbuf();
      };
      auto at = p.small_cvo_pattern.find('@');
      if (at == std::string::npos) {
        append_path(p.small_cvo_pattern);
      } else {
        const std::string prefix = p.small_cvo_pattern.substr(0, at);
        int nshard = 0;
        if (!absl::SimpleAtoi(p.small_cvo_pattern.substr(at + 1), &nshard) ||
            nshard <= 0) {
          LOG(ERROR) << "Bad small_cvo shard spec: " << p.small_cvo_pattern;
          return 1;
        }
        for (int i = 0; i < nshard; ++i) {
          append_path(absl::StrCat(prefix, "-",
                                    absl::Dec(i, absl::kZeroPad5),
                                    "-of-",
                                    absl::Dec(nshard, absl::kZeroPad5)));
        }
      }
      append_path(p.cvo_path);
      out.close();
      postprocess_input = p.merged_cvo_path;
    }

    LOG(INFO) << "Trio Stage 3 (" << p.role << "): postprocess_variants";
    std::vector<std::string> pp_args = {
        absl::StrCat("--infile=", postprocess_input),
        absl::StrCat("--ref=", ref_flag),
        absl::StrCat("--output_vcf_outfile=", p.output_vcf),
    };
    if (!p.output_gvcf.empty()) {
      pp_args.push_back(absl::StrCat("--gvcf_outfile=", p.output_gvcf));
    }
    auto argv_pp = MakeArgv("deepvariant_postprocess", pp_args);
    int n = static_cast<int>(argv_pp.size()) - 1;
    if (int rc = RunPostprocessVariants(n, argv_pp.data()); rc != 0) {
      LOG(ERROR) << "Trio: postprocess failed for " << p.role;
      return rc;
    }
    LOG(INFO) << "Trio: " << p.role << " VCF: " << p.output_vcf;
  }

  LOG(INFO) << "Trio: done. 3 VCFs at " << out_child << ", " << out_parent1
            << ", " << out_parent2;
  return 0;
}

// ──────────────────────────────────────────────────────────────────────
// DeepSomatic dispatch: one make_examples (tumor + optional normal),
// 1× call_variants on the tumor model only (normal has skip_output=true),
// 1× postprocess writing a single tumor VCF. Mirrors run_deepsomatic.py
// command sequence.
// ──────────────────────────────────────────────────────────────────────
int RunAllSomatic(int argc, char** argv) {
  const std::string ref_flag = absl::GetFlag(FLAGS_ref);
  const std::string regions_flag = absl::GetFlag(FLAGS_regions);
  const std::string tmp_dir = absl::GetFlag(FLAGS_intermediate_results_dir);
  const int num_shards = absl::GetFlag(FLAGS_num_shards);
  const int n_threads = std::max(1, num_shards);

  const std::string reads_tumor  = absl::GetFlag(FLAGS_reads_tumor);
  const std::string reads_normal = absl::GetFlag(FLAGS_reads_normal);
  if (reads_tumor.empty()) {
    LOG(ERROR) << "Somatic: --reads_tumor required";
    return 1;
  }
  const bool has_normal = !reads_normal.empty();

  const std::string out_vcf = absl::GetFlag(FLAGS_output_vcf);
  if (out_vcf.empty()) {
    LOG(ERROR) << "Somatic: --output_vcf required";
    return 1;
  }

  std::string ckpt = absl::GetFlag(FLAGS_checkpoint);
  if (ckpt.empty()) {
    LOG(ERROR) << "Somatic: --checkpoint (.dvw) required";
    return 1;
  }

  const std::string sm_path =
      absl::GetFlag(FLAGS_small_model_path_somatic);

  const std::string inference_backend =
      absl::GetFlag(FLAGS_inference_backend);

  // Per-stage intermediate paths.
  const std::string examples_pattern  =
      n_threads > 1
          ? absl::StrCat(tmp_dir, "/examples_tumor.tfrecord@", n_threads)
          : absl::StrCat(tmp_dir, "/examples_tumor.tfrecord");
  const std::string small_cvo_pattern =
      n_threads > 1
          ? absl::StrCat(tmp_dir, "/small_cvo_tumor.tfrecord@", n_threads)
          : absl::StrCat(tmp_dir, "/small_cvo_tumor.tfrecord");
  const std::string cvo_path        =
      absl::StrCat(tmp_dir, "/cvo_tumor.tfrecord");
  const std::string merged_cvo_path =
      absl::StrCat(tmp_dir, "/merged_cvo_tumor.tfrecord");

  // ── Stage 1: make_examples (tumor + optional normal). ────────────
  LOG(INFO) << "Somatic Stage 1: make_examples ("
            << (has_normal ? "tumor+normal" : "tumor-only")
            << ", --threads=" << n_threads << ")";
  {
    std::vector<std::string> me_args = {
        absl::StrCat("--reads_tumor=", reads_tumor),
        absl::StrCat("--ref=", ref_flag),
        absl::StrCat("--examples_tumor=", examples_pattern),
        absl::StrCat("--threads=", n_threads),
        "--task_id=0",
        "--num_shards=1",
        "--realigner_enabled=true",
    };
    if (has_normal) {
      me_args.push_back(absl::StrCat("--reads_normal=", reads_normal));
    }
    if (!regions_flag.empty()) {
      me_args.push_back(absl::StrCat("--regions=", regions_flag));
    }
    if (!absl::GetFlag(FLAGS_sample_name_tumor).empty()) {
      me_args.push_back(absl::StrCat("--sample_name_tumor=",
                                      absl::GetFlag(FLAGS_sample_name_tumor)));
    }
    if (!absl::GetFlag(FLAGS_sample_name_normal).empty()) {
      me_args.push_back(absl::StrCat("--sample_name_normal=",
                                      absl::GetFlag(FLAGS_sample_name_normal)));
    }
    if (!sm_path.empty()) {
      me_args.push_back(absl::StrCat("--small_model_path_somatic=", sm_path));
      me_args.push_back(absl::StrCat("--small_model_cvo_outfile_tumor=",
                                      small_cvo_pattern));
    }
    auto argv_me = MakeArgv("deepvariant_make_examples", me_args);
    int n = static_cast<int>(argv_me.size()) - 1;
    if (int rc = RunMakeExamples(n, argv_me.data()); rc != 0) {
      LOG(ERROR) << "Somatic: make_examples failed";
      return rc;
    }
  }

  // ── Stage 2: call_variants on the tumor model. ────────────
  LOG(INFO) << "Somatic Stage 2: call_variants";
  {
    // Somatic WGS pileup is 200×221×7 (tumor 100 + normal 100). For
    // tumor_only the height is 100. Pass via --input_height.
    const int tumor_h_default = has_normal ? 200 : 100;
    std::vector<std::string> cv_args = {
        absl::StrCat("--examples=", examples_pattern),
        absl::StrCat("--outfile=", cvo_path),
        absl::StrCat("--checkpoint=", ckpt),
        absl::StrCat("--batch_size=", absl::GetFlag(FLAGS_batch_size)),
        absl::StrCat("--inference_backend=", inference_backend),
        absl::StrCat("--input_height=", tumor_h_default),
        "--input_channels=7",
    };
    auto argv_cv = MakeArgv("deepvariant_call_variants", cv_args);
    int n = static_cast<int>(argv_cv.size()) - 1;
    if (int rc = RunCallVariants(n, argv_cv.data()); rc != 0) {
      LOG(ERROR) << "Somatic: call_variants failed";
      return rc;
    }
  }

  // ── Stage 2.5: merge small_cvo into cvo (if SM was used). ──
  LOG(INFO) << "Somatic Stage 2.5: merge → " << merged_cvo_path;
  {
    std::vector<std::string> cmd = {
        "/bin/sh", "-c",
        absl::StrCat("cat ", cvo_path, " > ", merged_cvo_path)};
    if (!sm_path.empty()) {
      // Pre-pend small_cvo records.
      cmd[2] = absl::StrCat(
          "cat ",
          n_threads > 1 ? absl::StrCat(tmp_dir, "/small_cvo_tumor.tfrecord-*")
                        : small_cvo_pattern,
          " ", cvo_path, " > ", merged_cvo_path);
    }
    int rc = std::system(cmd[2].c_str());
    if (rc != 0) {
      LOG(ERROR) << "Somatic: merge step failed";
      return 1;
    }
  }

  // ── Stage 3: postprocess. ────────────
  LOG(INFO) << "Somatic Stage 3: postprocess_variants";
  {
    std::vector<std::string> pp_args = {
        absl::StrCat("--ref=", ref_flag),
        absl::StrCat("--infile=", merged_cvo_path),
        absl::StrCat("--output_vcf_outfile=", out_vcf),
        "--process_somatic=true",
    };
    auto argv_pp = MakeArgv("deepvariant_postprocess", pp_args);
    int n = static_cast<int>(argv_pp.size()) - 1;
    if (int rc = RunPostprocessVariants(n, argv_pp.data()); rc != 0) {
      LOG(ERROR) << "Somatic: postprocess_variants failed";
      return rc;
    }
  }

  LOG(INFO) << "Somatic: done. VCF at " << out_vcf;
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
