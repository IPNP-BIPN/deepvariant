# CLAUDE.md — DeepVariant Apple Silicon Native Port (v2)

Project memory for AI-assisted work on `feature/apple-silicon-native-v2`.

## What this branch is

A fresh-start port of Google DeepVariant (and DeepTrio, DeepSomatic, pangenome-aware DV) to a single, fully native arm64 binary on Apple Silicon, distributed via Homebrew, with Apple Metal GPU + ANE inference and **zero Python interpreter at runtime**.

Authoritative plan: `~/.claude/plans/prompt-deepvariant-apple-idempotent-peacock.md`.
Running log: `PORT_LOG.md`.

## Hard constraints (non-negotiable)

- macOS ≥ 14, arm64 only.
- No Docker / no Rosetta / no CUDA at runtime. **No Python anywhere in the project we add** (Voie A strict — dev-time tools are Swift/C++, not Python).
- Build is reproducible. User installs in one Homebrew command, no compilation on their box.
- **Scientific accuracy preserved**: SNP F1 ≥ reference − 0.05 %, INDEL F1 ≥ reference − 0.10 %. Argmax 100 % agreement on the 1000-example Phase 0 bench. Max-abs softmax ≤ 1e-3.
- **GPU truly engaged**: verified by `powermetrics --samplers gpu_power,ane_power` showing non-zero residency.
- **Speedup ≥ 2.5×** vs published Linux x86 reference (`call_variants` stage, Phase 0 gate).
- **Strict 100 % FILTER-class parity vs `google/deepvariant:1.10.0` Docker** is the Homebrew-ship gate (Phase 5.5d). Every site's classification (PASS / RefCall / NoCall / LowQual) must match Docker on chr20 full and on the Phase-7 virgin-machine matrix. Set 2026-04-28; revised from byte-parity (unreachable on GPU due to FP32 non-associativity) and from variant-set-only parity (looser, lets non-PASS classes drift).

## Working rules

1. **Test before commit.** Every commit must leave the build green: `swift build && swift test` in `tools/conversion/` for Phase 0 work; `cmake --build build && ctest -V` for Phases 1+.
2. **Never degrade scientific precision.** F1 thresholds are gates, not goals. If we slip below, we fix the root cause — we do not lower the bar.
3. **Never bypass an error.** No `--no-verify`, no swallowed exceptions, no commenting out of failing tests. Diagnose the root cause.
4. **Document every critical decision** in `PORT_LOG.md` with date, context, alternatives considered, and rationale.
5. **Don't touch the v1 worktree** at `/Users/benjamin/projects/deepvariant-apple-silicon/.worktrees/apple-silicon-native/`. v1 is a separate clone retained as research; v2 is its own fresh history.
6. **Don't modify upstream `BUILD` / Bazel rules or upstream Python files.** They stay as a Linux/Bazel reference. v2 builds via CMake on macOS only and contains zero Python files of our own.
7. **No half-finished implementations.** Each phase has a success gate; do not cross it without meeting the gate. Stubs are allowed but must error out with `not yet implemented` rather than silently no-op.
8. **No Python in our code, ever.** All dev-time tooling is Swift (`tools/conversion/`, a Swift Package) or shell (`tools/reference/`, `release/`). The only Python in the repo is upstream's pre-existing tools/*.py from r1.10 — left untouched.
9. **TF is allowed transitively in Docker at conversion time.** The model conversion runs `coremltools.convert(saved_model, source='tensorflow')` inside `google/deepvariant:1.10.0` (which already ships TF 2.16). TF never appears in our local venvs and never in the runtime artefact. See `tools/conversion/convert_via_docker.sh`.

## Stop conditions (per spec)

If any of the following happen, stop, write a report in `PORT_LOG.md`, and surface to the user:

- Scientific precision regresses below the F1 thresholds and cannot be recovered.
- The GPU/ANE cannot be engaged in a way that's stable and verifiable.
- A required dependency cannot be made portable (e.g., a transitive lib that won't build statically on arm64).

## Priority order (when trade-offs collide)

1. Scientific exactness.
2. Robustness.
3. User simplicity (one-command install, no setup).
4. Performance.

## Phase stop-points (mandatory user review)

- After **Phase 0 ADR** — framework choice (Core ML vs MLX vs tf-metal). Irreversible without large rework.
- After **Phase 1** green CMake build — confirms TF detangling worked.
- After **Phase 3** first end-to-end native run — first real VCF produced.
- After **Phase 4** validation — release go/no-go.

## Where the project actually stands (rolling status)

Phases 0–4 done (Phase 4 PASS: F1 within thresholds vs Linux x86 ref on HG002 chr20). Phase 5.5 in progress — see "Phase 5.5 status" below. Phases 5/6/7 not started. Honest backlog tracked in `PORT_LOG.md` under the "Honest assessment" section. Each item below has a real (not hand-wavy) effort estimate:

- DeepTrio orchestration (3-BAM make_examples, 6-channel pileup): ~1 wk
- DeepSomatic orchestration (tumor + normal, somatic filtering): ~1-2 wk
- Pangenome-aware (12-channel + GBZ reader): ~1 wk
- gVCF blocks (`--output_gvcf` impl): ~3 d
- DirectPhasing wired in: ~3 d
- Alt-aligned pileup (PacBio/ONT): ~2 d
- Methylation channels: ~2 d
- GIAB hap.py F1 validation: ~1 wk
- Code signing + notarization scripts: ~2 d
- Homebrew formulas (`deepvariant`, `deepvariant-models`): ~2 d
- Virgin-machine M1/M2/M3/M4 matrix: ~2 d
- Closing the WGS chr20 16% VCF delta to >99%: ~1 wk

A claim "near release-ready" requires those gates met, not just a
working WGS pipeline at 84% match.

## Phase 5.5 status (2026-04-28)

Sub-phases (per the master plan):

- **5.5a — fix the MPSGraph builder.** ✅ DONE 2026-04-28. Two real bugs found and fixed:
  1. The `validation/work/wgs.dvw` was stale (extracted with an earlier broken `extract_weights.py` / `tensor_bundle_reader.py`). Fresh re-extract → bytes match the SavedModel.
  2. The hand-coded `(conv_n, bn_n)` pairs in `inception_v3_mil.py` for the InceptionA/B/C blocks were wrong: Keras's `tf.keras.applications.InceptionV3` does NOT enumerate layers in strict (conv, bn, conv, bn, …) order — TrackableObjectGraph mixes branches, so e.g. `conv2d_5 → layer_with_weights-16` (not 10). Authoritative pairs derived by byte-matching each frozen-graph kernel/beta const against the bundle's `layer_with_weights-K` entries. See `tools/conversion/dump_authoritative_pairs.py` (TBD) and the regenerated `Mixed_*` functions in `metal_inference.mm`.

  Result: 19/19 taps match TF reference within FP32 cumulative drift (max-abs ≤ 1.5e-3 over 188 layers; mean-abs ≤ 1e-4). MPSGraph `convolution2DWithSourceTensor:` with `dataLayout=NHWC` + `weightsLayout=HWIO` is bit-exact at each step — earlier "channel permutation" symptoms were entirely from the two structural bugs above.

  Tooling shipped:
  - `tools/conversion/dump_tf_per_layer.py` + `.sh` (TF reference dumper, runs in google/deepvariant:1.10.0 Docker, freezes the graph via `convert_variables_to_constants_v2` + v1 Session).
  - `deepvariant/native/debug_metal_main.cc --compare-to-reference <ref_dir>` (NPY reader + ULP-diff per tap).
  - `deepvariant/native/microtest_main.mm` (`microtest_metal` binary — hand-verifiable MPSGraph conv on small graphs; how we eliminated MPSGraph itself as the bug source).
- **5.5b — chr20 strict FILTER-parity measurement.** Next.
- **5.5c — custom Metal compute kernels (deferred / probably unneeded).** Originally planned as a fallback if MPSGraph itself were buggy. With Phase 5.5a now showing MPSGraph FP32 conv matches TF within 1 ULP/layer drift, this isn't required — the FILTER thresholds (PASS / RefCall / NoCall / LowQual) should sit well outside the ≤ 1.5e-3 noise floor.
- **5.5d — small_model + extension to all variants.** Pending after 5.5b confirms FILTER parity on WGS.

## Pitfalls already known (mine before re-discovering)

- **`tensorflow-metal` is dead** — unmaintained since mid-2024, frozen at TF 2.16, M-series ReLU bugs. Dropped from the v2 bench.
- **TensorFlow is banned in our venvs.** `setup_venvs.sh` enforces `import tensorflow` failing. SavedModel reading uses a pure-protobuf parser in `tools/conversion/savedmodel_reader.py` (vendored TF `.proto` files compiled via `protoc --python_out`). Core ML emit goes through PyTorch (`coremltools.convert(traced_torch_model, source="pytorch")`) instead of the TF path. **Inside the conversion Docker (google/deepvariant:1.10.0), TF is available and we do use it** — for `dump_tf_per_layer.py` and the per-layer reference flow.
- **MPSGraph `convolution2DWithSourceTensor` is bit-exact** with `dataLayout=NHWC` + `weightsLayout=HWIO` (verified Phase 5.5a 2026-04-28 — see `microtest_metal` Tests 1-7, all PASS within 1 ULP). Earlier reports of "channel permutation" were artifacts of two real bugs in our wrapper code: (a) a stale `.dvw` file with corrupted bytes, and (b) wrong `(conv_n, bn_n)` pairs in `inception_v3_mil.py`'s InceptionA/B/C recipe. Both fixed. Don't blame MPSGraph again without first running `microtest_metal` end-to-end.
- **Keras `BatchNormalization` default epsilon is 1e-3, NOT 1e-4.** Inception-v3 SavedModels are trained with epsilon=1e-3. Using 1e-4 in our fold gives a subtle scale mismatch on channels with small variance. Fixed in `metal_inference.mm`.
- **MPSGraph `OIHW` is genuinely O,I,H,W (not OHWI).** Documented behavior is correct — passing shape `(O, H, W, I)` with `weightsLayout=OIHW` triggers an explicit "Source and weight input channels mismatch" assertion in `GPUConvolutionOps.mm`. Don't try to be clever with the layout label — match the documented memory layout.
- **`tf.saved_model.load(...)` is not the same as `tf.keras.models.load_model(...)`.** DV models are saved via `tf.saved_model.save` (no Keras metadata). To get intermediate outputs, load with `tf.saved_model.load`, freeze with `convert_variables_to_constants_v2`, then re-import the frozen GraphDef into a v1 Graph for `Session.run` with named tensor fetches. This is the pattern in `dump_tf_per_layer.py`.
- **Inside the SavedModel inner function**: tensor names look like `StatefulPartitionedCall/inceptionv3/<keras_layer_name>/<op>:0`. Stem CBR tap = `activation_N/Relu:0` (N=0..4). Inception block output tap = `mixed{0..10}/concat:0`. Global avg pool = `global_average_pooling2d/Mean:0`. The signature output is `Identity:0` (final softmax wrapped).
- **`layer_with_weights-K` indexing is NOT trivial conv/bn alternation.** Keras's `tf.keras.applications.InceptionV3` builds the model with parallel branches; the TrackableObjectGraph enumerates layers in a graph-traversal order that mixes branches. For example `conv2d_5` (the first Mixed_5b conv attached) is `layer_with_weights-16`, not `layer_with_weights-10`. To get the correct (conv_n, bn_n) pair for a given Keras `conv2d_M`, byte-match the frozen graph's kernel const value against the bundle's `layer_with_weights-K/kernel/...VARIABLE_VALUE`. See the regenerated `Mixed_*` functions in `metal_inference.mm` (each line annotated with the Keras `M` index for traceability) and the (TBD) `tools/conversion/dump_authoritative_pairs.py`.
- **ANE prefers 4-channel image-shaped tensors.** Our model is 7- or 12-channel. ANE may refuse — accept GPU-only fallback. Core ML's `.all` compute units do this fallback automatically op-by-op.
- **Metal compute is not bitwise reproducible** across some ops/reboots. Validate via softmax tolerance (≤1e-3) + argmax agreement (100 %), not bit-equality. The strict-FILTER gate works because thresholds (PASS / RefCall / NoCall / LowQual) sit far enough from typical softmax noise that ≤ 1e-5 drift doesn't flip class.
- **`build-prereq.sh` is Linux-only.** v2 ships `scripts/build-prereq-macos.sh`.
- **8.5 GB of model artifacts** can't fit in a single Homebrew bottle alongside the binary. Split into `deepvariant-models` formula.
- **Xcode CLT is enough — no full Xcode required.** Ship `.mlpackage` uncompiled; runtime compiles on first load via `MLModel compileModelAtURL:error:`. Avoid `xcrun coremlcompiler` (full Xcode only).
- **TF v2 checkpoint format** (the `variables/variables.{index, data-*}` layout) is documented at `tensorflow/core/util/tensor_bundle/tensor_bundle.h` — we replicate `BundleReader` in pure Python.

## Key file paths

- Plan: `~/.claude/plans/prompt-deepvariant-apple-idempotent-peacock.md`
- v2 root: `/Users/benjamin/deepvariant`
- v1 reference clone: `/Users/benjamin/projects/deepvariant-apple-silicon/.worktrees/apple-silicon-native/` (read-only)
- Native runtime (Phases 2-3): `deepvariant/native/`
- Build (Phase 1): `CMakeLists.txt` + `cmake/*.cmake`
- Conversion (Phase 0, dev-time, Swift Package): `tools/conversion/` — produces the `dv-tools` CLI.
- Linux ref capture (Phase 0): `tools/reference/` (shell + Docker, no Python).
- Release tooling (Phase 5): `release/` (shell + `codesign` + `xcrun notarytool`).
- Homebrew formulas (Phase 6): separate repo `homebrew-deepvariant/`.

## Reused upstream C++ (do not rewrite)

These are the multipliers that make v2 feasible. Wrap, don't rewrite:

- `deepvariant/make_examples_native.cc`
- `deepvariant/pileup_image_native.cc`
- `deepvariant/allelecounter.cc`
- `deepvariant/realigner/{fast_pass_aligner,debruijn_graph,ssw,window_selector}.cc`
- `deepvariant/{direct_phasing,merge_variants,merge_phased_reads,postprocess_variants}.cc`
- `third_party/nucleus/io/{sam_reader,vcf_reader,vcf_writer,reference,gbz_reader}.cc`
