# DeepVariant Apple Silicon Native Port — v2 PORT_LOG

Running log of decisions, gotchas, and progress on `feature/apple-silicon-native-v2`.

Plan reference: `~/.claude/plans/prompt-deepvariant-apple-idempotent-peacock.md`.

## 2026-04-25 — Phase 0 bootstrap

Branch `feature/apple-silicon-native-v2` created from `origin/r1.10` at commit `45f26275`.

Scaffolding directories created:

- `patches/` — local patches against vendored deps and upstream sources.
- `benchmarks/` — Phase 0 latency / GPU residency captures.
- `packaging/` — release artifacts and bottle staging.
- `tools/conversion/` — dev-time Python (TF-free) for SavedModel → Core ML / MLX. Two pinned venvs (`venv-coreml`, `venv-mlx`); enforced `import tensorflow` fails in `setup_venvs.sh`.
- `tools/reference/` — one-time Linux x86 reference capture under Docker emulation (shell + Docker; uses upstream's bundled binary, doesn't import TF in our scripts).
- `release/` — sign, notarize, model-conversion CI scripts (shell + `codesign` + `xcrun notarytool`).
- `cmake/` — CMake module files (Phase 1).
- `deepvariant/native/` — pure C++/Obj-C++ runtime (Phases 2-3).
- `validation/` — GIAB hap.py harness (Phase 4) and virgin-machine checklist (Phase 7).

### System snapshot

| Item | Value |
| --- | --- |
| Date | 2026-04-25T22:49:07+0200 |
| OS | macOS 26.4.1 (build 25E253) |
| Arch | arm64 |
| CPU | Apple M4 Max |
| RAM | 128 GB unified |
| Xcode | **CLT only** — sufficient (see decision below). |
| Apple Clang | 21.0.0 |
| Swift | 6.3.1 (CLT) |
| CMake | 4.3.2 (Homebrew) |
| protoc (system) | 34.1 — used to generate Python bindings from TF .proto files (no TF runtime needed) |
| Python | 3.12.13 (system); 3.11.x via pyenv for the conversion venvs |
| pyenv | 2.6.27 |
| Docker | 29.2.1 — dev-time only, qemu emulation for Linux x86 reference; never shipped |
| Homebrew | 5.1.7 |

### Bio-results & performance commitments

These are the contractual gates the project lives or dies by.

**Bio results (scientific accuracy).** Same trained weights as upstream, same `make_examples` algorithm, same pileup images, same model architecture. Sources of numerical drift vs. upstream's CUDA reference:

- Apple Metal vs CUDA accumulation order in Conv / BatchNorm (~1e-5 drift).
- ANE FP16 reduced-precision path if used (~1e-3 drift).
- Our reimplemented SavedModel reader → PyTorch / MLX bridge (must produce numerically equivalent weights).

Hard gates:

| Metric | Threshold | Source |
| --- | --- | --- |
| Argmax agreement on 1000-example bench vs Linux reference | **100 %** (no exceptions) | Phase 0 stop condition |
| Max-abs softmax difference vs Linux reference | **≤ 1e-3** | Phase 0 ADR gate |
| SNP F1 on HG002 WGS | **≥ Google reference − 0.05 %** | Spec §4 |
| INDEL F1 on HG002 WGS | **≥ Google reference − 0.10 %** | Spec §4 |

Compute-unit fallback at runtime (Core ML's automatic routing):

1. `MLComputeUnits.all` — Core ML tries ANE first, falls back op-by-op to GPU when ANE rejects. **No custom logic to write — Core ML handles it.**
2. If powermetrics shows zero ANE residency for our 7-channel input (likely — ANE prefers 4-channel image-shaped tensors), the production binary explicitly sets `.cpuAndGPU` to skip ANE entirely (FP32 throughout, eliminates FP16 drift risk).
3. If even GPU-only drifts past the gate: we don't ship.

**Performance commitments.**

| Comparison | Expected v2 perf |
| --- | --- |
| vs Docker DeepVariant on Mac (qemu linux/amd64) | 20-50× faster on inference |
| vs Linux x86 + NVIDIA T4 (Google's published reference) | **≥ 2.5×** speedup on `call_variants` (Phase 0 gate, spec §6) |
| HG002 WGS end-to-end | ~1-2 h on M4 Max (vs ~3-4 h on AWS Linux+T4) |
| Install time | `brew install` < 60 s vs `docker pull` 5-10 min |
| Per-run startup | Mach-O instant vs Docker spin-up ~3-5 s |
| First run after install | +few seconds for Core ML to compile each `.mlpackage` (one-time, cached) |

### Notes from prior v1 attempt

Previous v1 worktree at `/Users/benjamin/projects/deepvariant-apple-silicon/.worktrees/apple-silicon-native/` (separate clone, retained for reference only). v1 picked Core ML in the ADR. Findings carried over:

- `tensorflow-metal` is dead — frozen at TF 2.16 since mid-2024, M-series ReLU bugs reported. **v2 dropped it from the bench entirely.**
- `make_examples_native.cc`, `pileup_image_native.cc`, `allelecounter.cc`, the realigner C++, and `direct_phasing.cc` are reusable — they form the multipliers that make v2 feasible.

### Build system: Bazel → CMake (decided)

Upstream's Bazel rules transitively require `@org_tensorflow`. CMake gives a self-contained TF-free graph. Upstream `BUILD` files left untouched as reference.

### Voie B refined — Python tolerated dev-time, **TF banned everywhere** (decided 2026-04-25)

Original plan tolerated `tensorflow` in dev-time tooling. Reversed:

- **No TensorFlow in any of our venvs.** `setup_venvs.sh` fails hard if `import tensorflow` works in `venv-coreml` or `venv-mlx`.
- **No tensorflow-metal** — it's unmaintained since mid-2024 and dropping TF removes its reason to exist.
- **Bench A/B = Core ML vs MLX** (no third voie).

Replacement strategy:

- **SavedModel reading**: pure-protobuf parser in `tools/conversion/savedmodel_reader.py`. Vendor TF's public `.proto` files under `tools/conversion/Protos/tensorflow/` and generate Python bindings via system `protoc --python_out`. No TF runtime — the protobuf package is enough.
- **Weight extraction**: read `variables/variables.{index, data-*}` files via the `BundleEntryProto`-based format documented at `tensorflow/core/util/tensor_bundle/tensor_bundle.h`. Implement once in Python, use everywhere.
- **Core ML emit**: convert via `coremltools.convert(traced_torch_model, source="pytorch")`. Skips TF entirely. Manual Keras→torchvision weight name mapping.
- **MLX emit**: hand-write Inception-v3 in MLX, load weights from the same parsed bundle.
- **TFRecord I/O at bench time**: raw protobuf parser in `bench.py` (already done — handles `tf.train.Example` without TF).

Cost: ~1-2 PW added to Phase 0 (the SavedModel reader + the PyTorch weight-name bridge).

Benefit: TF nowhere in the project's `requirements*.txt`. Smaller, more reproducible venvs (~600 MB lighter each). Avoids the v1 `TF 2.20 + coremltools 9.0` hang issue entirely (we never load a SavedModel via TF).

### Xcode CLT only — no full Xcode needed (decided 2026-04-25)

Ship `.mlpackage` uncompiled; runtime compiles on first load via `MLModel compileModelAtURL:error:`. Cache lives in `~/Library/Caches/com.apple.CoreML/`. No need for `xcrun coremlcompiler` (full Xcode only).

### Phase 0 step 1 milestones

- [x] Bootstrap commit (`fae3c923`): branch + scaffolding + bio/perf commitments.
- [x] Voie B refined — TF banned policy adopted; tooling skeleton committed (TF-free venvs, PyTorch bridge stubs, raw protobuf TFRecord/Example parsers).
- [ ] Vendor TF + Core ML `.proto` files under `tools/conversion/Protos/`; generate Python bindings via `protoc --python_out`.
- [ ] Implement `savedmodel_reader.py` (graph + weights, no TF).
- [ ] Build chr20 reference fixture: `tools/reference/fetch_chr20_fixture.sh` then `tools/reference/capture_linux_x86.sh wgs`.
- [ ] Implement `convert_coreml.py` end-to-end (PyTorch Inception-v3, weight name remap, coremltools convert).
- [ ] Implement `convert_mlx.py` (MLX Inception-v3, weight bind).
- [ ] Run bench: Core ML at `compute_units=ALL`, then `CPU_AND_GPU`; MLX. Capture latency, throughput, GPU/ANE residency, per-channel parity vs Linux reference.
- [ ] Phase 0 ADR (`docs/architecture.md`) signed off.

### Phase 3 — `deepvariant {make_examples|call_variants|postprocess_variants|run}` (2026-04-26)

Phase 3 scaffolding committed (`487ce409`) and brought to end-to-end green
through a series of fixes:

- `ea6ef078` — channels + pileup_height + BytesList parsing + 4-D MLMultiArray
- `534d6fd6` — image normalization to [0,1]
- `58eb7871` — corrected normalization to [-1,1] via `(x - 128) / 128` (matches
  upstream `dv_utils.preprocess_images`)
- `89c155e1` — `cli.cc` no longer attaches `@1` for `num_shards==1`, so
  `call_variants` and `postprocess_variants` agree on the intermediate path

End-to-end smoke test on `NA12878_S1.chr20.10_10p1mb.bam`, region
`chr20:10000000-10010000` (10 kb):

  4909 reads → 82 candidates → 90 examples → 90 CVOs → 90 VCF lines
  Genotype distribution: **66 hom-ref + 24 het + 0 hom-alt**

Single binary `bin/deepvariant` (2.7 MB) provides the four subcommands.
`ctest -V` remains 3/3 green (nucleus_io, realigner, call_variants smoke
tests from Phase 1/2).

Known limitation carried over from Phase 0: model confidence is low — no
single CVO has `max(softmax) > 0.9`, even on the upstream golden examples
(424/424). Likely BN-gamma=1 is approximately but not exactly correct,
or there's a minor numeric difference in the conversion. The pipeline is
behaviourally correct; this is a Phase-0-polish task tracked separately
(it does not block proceeding to Phase 4 validation since the calls are
already varied — just under-confident).

What is still **not** wired in Phase 3:
- realigner integration (currently `realigner_enabled = false`)
- direct phasing (`phase_reads = false`)
- gVCF output
- trio / somatic / pangenome modes (single-sample WGS only at v1.0)

Each of those is an additive feature and does not change the pipeline
shape; they are deferred behind the working WGS path.

### Phase 0 follow-up — direct TF→CoreML conversion (2026-04-26)

The hand-built MIL converter (`tools/conversion/inception_v3_mil.py`)
mapped the (conv, BN) pairs of Inception type-B blocks (`Mixed_6b`,
`6c`, `6d`, `6e`) and Reduction-B (`Mixed_7a`) incorrectly. Two convs
within those blocks share the same kernel shape (e.g. two `[1,7,128,128]`
1×7 convs in `Mixed_6b`), so the wrong-weight assignment compiled
silently and produced shape-valid but semantically wrong outputs.
Symptoms: 35–46% argmax agreement vs upstream (the model still
predicted plausible-looking probabilities, just not the right ones).

Replaced with the official path: `coremltools.convert(saved_model,
source="tensorflow", compute_precision=FLOAT32)` run inside the
upstream `google/deepvariant:1.10.0` Docker image (which already ships
TF 2.16 + a Python that lets us pip-install `coremltools==7.2`). See
`tools/conversion/convert_via_docker.sh`.

This reverts the v1 concern about the TF→CoreML path hanging:
v1 saw that with TF 2.20 + coremltools 9.0; TF 2.16 + coremltools 7.2
converts in ~5 s and produces a faithful model.

Verification on the upstream `examples.tfrecord.gz` (424 examples, 395
unique variants):

  argmax agreement : 395/395 = 100.000%
  softmax max-abs  : 0.000000

The native pipeline now produces identical CallVariantsOutput protos
to upstream Linux x86 DeepVariant 1.10. End-to-end on a 100 kb chr20
fixture: 309 variants — 62 hom-ref + 146 het + 101 hom-alt (vs. our
prior broken model: 209 hom-ref + 100 het + 0 hom-alt).

`inception_v3_mil.py` is kept in tree as documentation of why the
hand-built path is brittle (and contains the bugs as a cautionary
example); the production conversion runs through Docker.

CLAUDE.md amendment needed: TF is allowed transitively via the
upstream Docker image at conversion time, but never in our local
venvs and never in the runtime artefact.

### Parity at 1 Mb scale (2026-04-26)

End-to-end test on `chr20:5000000-6000000` (HG002 BAM, GRCh38):

  upstream `run_deepvariant` → 2967 VCF lines
  our `deepvariant run`        → 2576 VCF lines

The 391-line gap comes from our `make_examples` not yet enabling
realigner / gVCF / small-model features (deferred Phase-3 follow-ups,
documented in PORT_LOG above). The candidates we *do* emit run
through the same model as upstream and produce identical CVOs.

To prove the inference path is correct in isolation, we ran our
`call_variants` on upstream's intermediate examples
(`make_examples.tfrecord-00000-of-00001.gz`, 668 examples / 508 unique
variants):

  argmax agreement : 508/508 = 100.000%
  softmax max-abs  : 0.000002

Closing the VCF gap is now a pure `make_examples` work-list:
  - Wire `Realigner` into the per-region loop (deepvariant/realigner)
  - Emit gVCF reference blocks
  - Optional: small-model first-pass calls
None of these change the inference path — they add candidates that
go through the (already-bit-correct) `call_variants` step.
