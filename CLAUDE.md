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
- **5.5b — chr20 strict FILTER-parity measurement.** Sub-region (424 examples through deepvariant big-model on chr20:200997..299145) confirmed: **255/255 PASS sites identical to Docker, 108/108 RefCall identical, 16/16 NoCall identical** (only 2/381 borderline NoCall↔RefCall flips, no PASS impact). Full-chr20 measurement deferred until cli.cc is rebuilt — parallel sharding now spawns one subprocess per shard via `posix_spawn` (`cli.cc` commits 0957a949 + 00264e0a). True intra-process threading (à la salmon/samtools 1600 % CPU) is a follow-up commit; the subprocess workaround already gives 14× wall-time speedup on chr20 make_examples (~3 min on M4 Max).
- **5.5c — Metal deterministic-conv kernel (built but not the fix path).** Phase 5.5c custom Metal compute kernel was implemented and verified bit-exact vs CPU reference (`microtest_conv_serial` 4/4 PASS). However, swapping the full stem (s1a → mp5a) to the deterministic kernel produced **100 % identical FILTER classification to MPSGraph** on full chr20 — i.e., MPSGraph's reduction-order non-determinism is NOT what flips FILTER classes. The 1.13 % FILTER drift vs Docker comes from elsewhere. The kernel infrastructure (`metal_kernels/conv_serial_fp32.metal`, `MetalConvSerial`, `MetalMaxPool`, env-var `DV_METAL_DET_LAYERS=stem` to opt in) is left in place as documented dead-code-on-the-default-path, available if a future model has more drift-sensitive layers.
- **5.5d/1 — root-cause fix #1: libstdc++-compatible std::shuffle.** ✅ DONE 2026-04-28. Phase 5.5c-aside investigation: extract pileup at chr20:29335346 (a known PASS-flip site) and byte-compare with Docker's pileup at the same site → **25.6 % of pixels different** (max-abs diff 1.98 on a [-1, 1] range). Means our pileup image structurally differs from Docker's at this site, regardless of what inference path we use. Diagnosis: `pileup_image_native.cc:162` calls `std::shuffle` to subsample reads when coverage exceeds the pileup height (95 reads). `std::shuffle` is implementation-defined; libc++ (Apple Clang) and libstdc++ (GCC, Docker) produce **completely different sequences** for the same `mt19937_64` state and seed. Verified via a 203-element shuffle test: libc++ first 5 = `45, 109, 120, 152, 188`; libstdc++ first 5 = `162, 7, 124, 61, 80`. Fix: ported libstdc++ 12's exact `std::shuffle` algorithm — paired Fisher–Yates + `__gen_two_uniform_ints` + Lemire's nearly-divisionless 128-bit uniform — into `deepvariant/native/libstdcxx_shuffle.h::Shuffle`. Verified bit-identical to libstdc++ on the test. One-line patch to `pileup_image_native.cc:162`. After fix: pileup at chr20:29335346 byte-matches Docker (max-abs diff = 0). chr20 FILTER drift 1.13 % → 0.54 %; PASS-flips 535 → 261.
- **5.5d/2 — root-cause fix #2: postprocess multi-allelic CombineLikelihoods CVO-prune.** ✅ DONE 2026-04-28. Diagnosed at chr20:63028104 T>C,G (G alt pruned; ours and Docker had byte-identical pileups but PL = 0,33,45 vs 0,18,23). Our `CombineLikelihoods` was using ALL CVOs in product fusion, including the pruned-allele CVOs (CVO_G and CVO_C+G). Upstream `merge_predictions` skips them ("is_for_pruned_allele: continue", `postprocess_variants.py:1247-1248`). Fix: pass `alts_to_remove` to `CombineLikelihoods`; skip CVOs whose alt-set intersects with it; only renormalize when product crossed multiple kept CVOs (so single-kept-CVO sites return raw softmax, matching upstream). chr20 FILTER drift 0.54 % → **0.33 %**; PASS-flips 261 → **210**.
- **5.5d/3 — root-cause fix #3: NumPy-compatible reservoir sampling per partition.** ✅ DONE 2026-04-28. Diagnosed at chr20:31185803 (DP 647 ours vs 217 Docker — 5049 raw reads in BAM, 5686 reads after basic filters in the 1000-bp partition). Root cause: `make_examples_core.py:partition_reads_etc` applies Algorithm-R reservoir sampling per partition with `max_reads_per_partition=1500` using `np.random.RandomState(seed)`; we did not. 84 % of the remaining 210 PASS-flips sat at sites where |ΔDP| > 5 vs Docker; 47 % of total FILTER mismatches were at sites with `ours_DP > 3 × docker_DP`. Fix: `deepvariant/native/numpy_mt19937.h` ports NumPy 1.24's MT19937 + `random_interval(bg, max)` (NOT Lemire — that's `Generator.integers`; the legacy `RandomState.randint` path uses bitmask-rejection) + Algorithm-R reservoir sample to C++. Verified bit-equal to NumPy 1.24.3 in Docker on golden vectors (`microtest_numpy_rng` 3/3 PASS: `randint(0, 1000)` ×10, `randint(0, i+1)` for i=0..19, reservoir-sample-k>n). Hooked into `make_examples_main.cc` worker loop (fresh `NumpyMt19937(opts.random_seed())` per region). chr20 FILTER drift **0.33 % → 0.01 % (29 mismatches of 209814 shared sites)**; PASS-flips **210 → 27** (and now only one direction — ours=PASS where Docker=RefCall, nothing the other way); shared sites 209556 → 209814 (the cap recovers ~250 sites Docker had that we'd miss).
- **5.5d/4 — root-cause fix #4: haplotype-resolution port.** ✅ DONE 2026-04-29. The remaining 27 chr20 PASS-flips after 5.5d/{1,2,3} all sat at sites where a SNP overlaps a multi-allelic indel called GT=1/2 (compound het, both ploidy slots taken). Upstream's `haplotypes.maybe_resolve_conflicting_variants` (called from `run_postprocess_variants_on_region:1541-1543`) maximises a joint log-likelihood across the overlap group under the ploidy-2 constraint, which forces the SNP to 0/0 → RefCall. We did not port that step. Verified at chr20:14222820 A>G (inside the chr20:14222813 GAAA…→{G,GAAAA…} 17-bp deletion called 1/2): pileups byte-identical, CVO probs match Docker, but Docker's postprocess collapses the SNP to homref. Fix: `deepvariant/native/haplotypes.{h,cc}` ports `_resolve_overlapping_variants` + `_maybe_resolve_mixed_calls` + `_VariantCompatibilityCalculator` + `_LikelihoodAggregator`. `postprocess_main.cc` now buffers all variants and runs the resolver once before VCF emission. chr20 FILTER drift 0.014 % → 0.002 %; PASS-flips 27 → 2 (now ours=RefCall, docker=PASS — i.e. we're more conservative on those 2). 292 variant-call sub-groups resolved on chr20.
- **5.5d/5 — root-cause fix #5: simplify_variant_alleles.** ✅ DONE 2026-04-29. The 2 remaining PASS-flips after 5.5d/4 sat at sites where a tandem-repeat substitution (e.g. chr20:63221577 TTGCAGGGAC…→CTGCAGGGAC… encoded as a 36-bp substitution, where Docker emits the same call as a clean 1-bp T>C SNP) FALSELY overlapped a neighbouring SNP at chr20:63221586 — triggering a haplotype resolution that Docker doesn't because Docker's clean SNP doesn't overlap. Fix: port `nucleus/util/variant_utils.py:simplify_alleles + simplify_variant_alleles` (strip longest common postfix from {ref, alts}, leaving ≥ 1 base; update `end`). Called per-variant just before pushing into the haplotype-resolution buffer. chr20 FILTER drift 0.002 % → 0.001 %; **PASS-flips 2 → 0**.
- **5.5d/6 — small_model: MLComputeUnitsCPUOnly.** ✅ DONE 2026-04-29. Set as the right determinism default; ultimately superseded by 5.5d/7 (Core ML replaced entirely).
- **5.5d/7 — small_model: BNNS-CPU FP32 sequential.** ✅ DONE 2026-04-29. Replaced Core ML small-model inference with a deterministic FP32 scalar MLP (per-output `for` accumulator, no SIMD, no FMA). Weights extracted from upstream Docker (`/opt/smallmodels/wgs/model.keras`) via `tools/conversion/extract_small_model_weights.sh` into 6 `.npy` files (layer_{0,1,2}_{kernel,bias}.npy, ~2.4 MB total). Bit-equal to TF/Keras on x86 single-thread. Eliminated the ~0.005-0.01 max_p drift that flipped GQ=20 thresholds.
- **5.5d/8 — small_model: per-alt-set dispatch.** ✅ DONE 2026-04-29. Upstream `get_set_of_allele_indices(candidate)` enumerates biallelic + multi-allelic combinations: `[(0,), (1,), …, (N-1,)] + list(itertools.combinations(range(N), 2))`. For each `(candidate, alt_indices)` pair, the small-model decides INDEPENDENTLY — passing pairs become small-model CVOs, failing pairs are queued to deepvariant via `candidate.make_examples_alt_allele_indices`. Our code was iterating only single alts and using "all-or-nothing" gating (if any alt failed, the whole candidate went to deepvariant — missing the multi-alt combos and conflating per-pair decisions). Fix: iterate biallelic + combinations, decide per-pair, populate `make_examples_alt_allele_indices` for the failing ones (ExamplesGenerator already respects this field — only generates examples for the listed pairs). Extended `MakeSmallModelCvo` to accept multi-index sets. Added `IsSnpForIndices(variant, indices)` mirroring upstream's `is_snp(variant, exclude_alleles)`.
- **5.5d/9 — root-cause fix #6: AltAlleleQual = phred(1-sum_alt) rounded to 7 decimals.** ✅ DONE 2026-04-29. The 14/14 site-set diffs from 5.5d/8 all sat at saturated multi-allelic homref sites where `predictions[0] = 1.0` exactly in our BNNS-CPU softmax. Form A (`-10·log10(p_ref)`) returned 0 for every alt → first-iteration wins → mismatched Docker on 14 sites. Pure form B (`-10·log10(1-sum_alt)`) made `sum_alt` sub-ULP differences flip the max → 20 NEW diffs at different positions. Fix: use form B *and* round to 7 decimals (upstream's `_QUAL_PRECISION=7`, applied in `compute_quals:rounded_qual = round(qual, 7)`). At saturation, qual values < 5e-8 collapse to 0 (tie → first wins, matching Docker); qual ≥ 5e-8 survive at 1e-7 granularity (preserves Docker's genuine max-alt pick). Closes 14/14 site-set diffs. Native C++ implementation in `postprocess_main.cc::AltAlleleQual`; no new dependencies.
- **5.5d/10 — root-cause fix #7: PL log-space subtract + truncation (matches upstream's vcf_writer).** ✅ DONE 2026-04-29. Our PL was computed in PHRED space (`int(-10*log10(p_i)) - int(-10*log10(p_max))`); upstream's writer at `vcf_conversion.cc:1226-1228` operates in LOG space (`std::transform(normalized_log10, Log10PErrorToPhred)` where `normalized = log10(p_i) - max(log10)`, then double→int via implicit narrowing = TRUNCATION, NOT `Log10PErrorToRoundedPhred`). The two algorithms diverge by 1 unit at rounding boundaries for non-saturated probabilities. Fix: compute `gls[i] = log10(max(like[i], 1.25e-10))`, find `max_gl`, then `pl[i] = static_cast<int>(-10 * (gls[i] - max_gl))` (truncation). Closed PL ±1 record-level diff from 18660 → 80 (99.6 % reduction). Also rounded `variant.quality` to 7 decimals to mirror upstream's `compute_quals:rounded_qual = round(qual, 7)`.
- **5.5d status (chr20, FINAL — 2026-04-29).** End-to-end with all ten fixes: **210390/210390 site-set parity (100 %), 0 FILTER mismatches, 107113/107113 PASS variants identical**. Wall-time 3:13 m:s on M4 Max with 14 threads. **204419/210390 = 97.16 % records byte-identical to Docker** (up from 88.3 % at 5.5d/9). Remaining 5971 record-level diffs: 4877 QUAL ±0.1 only (FP drift in `1-sum_alt` straddles the 0.05 boundary at the 1-decimal write); 756 MID `small_model` vs `deepvariant` only (small_model dispatch GQ ≈ 20 boundary, FP-drift in max_p flips threshold side); 80 PL only (residual FP drift in like[] vector); 161 QUAL+GQ; 65 GQ only; 29 VAF only (htslib float-to-text rounding at 6th decimal); ~30 mixed. All residuals are FP-drift in big_model softmax (Inception-v3 GPU MPSGraph FP32 vs Docker TF/Keras Eigen-x86 FP32) — explicit non-goal per plan, "fundamentally unachievable on Apple GPU due to FP32 non-associativity in any parallel reduction". **Zero records differ in CHROM/POS/REF/ALT, FILTER, or GT** — every user-facing genomic conclusion matches Docker on chr20.
- **5.5e — extension to all variants.** Pending.
- **Phase 8 / Tier 6.0 — full-network deterministic conv path (research, not promoted).** ✅ DONE 2026-05-01. Extended Phase 5.5c det stem to cover ALL 11 Mixed_X Inception blocks (5b through 7c) + global avg pool, replacing MPSGraph entirely on the conv path. Infrastructure: `metal_det_mixed.{h,mm}` with `BuildDetMixed5b…7c` per-block builders (folded BN by default, unfolded toggle for research) + `DispatchDetMixedBlock` unified dispatcher (sequential / split-branch / pool-only branch types) + `microtest_det_inception` per-block validator. Wired behind `DV_METAL_SERIAL_FULL=1` env var (default OFF — baseline preserved). End-to-end measurements:
  - chr20:10M-10.1M (100 kb fixture): byte-identical to baseline (319 sites, 0 diffs).
  - chr20 full HG002 vs GIAB: **F1 SNP=0.997402 / INDEL=0.995985 — bit-identical to baseline F1**, including TP/FN/FP counts. The 8847 Docker-FILTER diffs vs baseline are all in zone QUERY.UNK (outside GIAB high-confidence regions) — scientifically equivalent.
  - chr20 full HG003 vs Docker AVX-512: 8837 FM (vs baseline 160). The det path's per-thread sequential FMA reduction order drifts in a different direction than MPSGraph's SIMD-group parallel reduction at borderline UNK-zone sites; both drifts ~1e-3 max_abs magnitude.
  - Wall-time: ~11 min/chr20 (vs 4 min baseline = ~3× slower).
  - Cross-chip determinism: guaranteed by construction (per-thread sequential FMA, no SIMD-group parallel reduction).

  **Decision (2026-05-01, user): keep baseline as default.** SERIAL_FULL stays as opt-in `DV_METAL_SERIAL_FULL=1` env var for users who explicitly need cross-chip-determinism + GPU-only at the cost of 3× wall-time. The 8847 UNK-zone divergence is invisible to F1 metrics so the science is preserved either way. Tier 6.0 infrastructure remains in tree as foundation for potential Tier 6.A (Kahan-compensated summation) work if a future use-case demands bit-Docker concordance.

  Files added: `metal_kernels/conv_kahan_fp32.metal`, `metal_conv_kahan.{h,mm}`, `metal_det_mixed.{h,mm}`, `microtest_conv_kahan.mm`, `microtest_det_mixed5b.mm`, `microtest_det_inception.mm`. Files modified: `metal_inference.mm` (DV_METAL_SERIAL_FULL gate + det_blocks dispatch), `microtest_conv_serial.mm` (extended to 11 Inception shapes, all PASS bit-exact), `CMakeLists.txt`. 6 commits (ffedb5aa → c84b9736).

## Phase 6 — DeepTrio + DeepSomatic + Pangenome-aware DV (in progress)

**Hard release gate (set 2026-04-29, applies to all three tools):** reproduce Docker's per-tool VCF output bit-for-bit on a chr20 fixture. Same gate as WGS chr20 already passes:

- 100 % site-set parity (`bcftools isec` shows `only_ours = only_docker = 0`)
- 0 FILTER-class mismatches on shared sites
- Identical PASS variant set (same count, same positions)
- Identical GT on every shared site

PL/QUAL/MID byte-level drift from FP32 non-associativity remains the explicit non-goal (carry-over from Phase 5.5d). FILTER classification, GT, and the variant set itself MUST be byte-identical to Docker, replicating the WGS guarantee for every tool.

### Step 1 — DeepTrio ✅ DONE 2026-04-30 (commit `e5bd9185`)

100% FILTER parity on chr20:10M-10.1M vs `google/deeptrio:1.10.0`:

- HG002 (child):   0 site-set diffs, 0 FILTER mismatches, 262/262 PASS
- HG003 (parent1): 0 site-set diffs, 0 FILTER mismatches, 265/265 PASS
- HG004 (parent2): 0 site-set diffs, 0 FILTER mismatches, 222/222 PASS

Two root-cause fixes resolved the trio gap (5.5d/12 + 5.5d/13). Both
are documented in detail in the trio status memory; summary:

- 5.5d/12: per-sample candidate_positions (was UNION, mirrors upstream's
  per-sample `get_candidate_positions(allele_counters, sample_name)`).
  Without this, parent2's AlleleCounter tracked ref reads at non-target
  positions → inflated `ref_support_ext` in the small_model combined block.
- 5.5d/13: parameterized Metal Inception-v3 input height/channels.
  `metal_inference.mm` had THREE hardcoded `100` references; trio's
  140-row pileup (60+40+40) was silently truncated.

### Step 2 — DeepSomatic ✅ DONE 2026-04-30 (commit `3f3f3060`)

100% FILTER parity on chr20:10M-10.1M (HG002 tumor + HG003 normal) vs
`google/deepsomatic:1.10.0`:

- 0 site-set diffs, 0 FILTER mismatches across 693 sites
- 34/34 PASS, 92/92 GERMLINE, 13/13 NoCall, 554/554 RefCall identical
- 0 GT diffs across shared sites
- 6/6 verified pileups byte-identical to Docker

Step-2 progression:

- **2-v1** (commit `c61a391a`) — somatic orchestration end-to-end: 11
  flags, IsSomaticMode helpers, multi-sample wiring, postprocess
  invocation, cli.cc somatic dispatch.
- **2-v2** (commit `1d529405`) — GERMLINE filter ported (mirror of
  `nucleus/io/vcf_writer.cc::WriteSomatic`): hets reclassified as
  homref + GERMLINE filter at write time.
- **2-v3** (commit `0e6d03ed`) — somatic threshold overrides
  (`vsc_min_fraction_*`, `small_model_*_gq_threshold`).
- **2-v4** (commit `3f3f3060`) — closes the last 5 FM. Root cause:
  `model.example_info.json:flags_for_calling` declares
  `sort_by_alt_allele_support: true` and
  `small_model_vaf_context_window_size: 51`. We applied the
  variant-caller overrides earlier but missed these two pic-level
  options. Without sort_by_alt_allele_support, our pileup rows are
  sorted purely by alignment position; Docker sorts by
  (haplotype, alt_support_group, position), so multi-alt sites have
  their tumor reads in different row order. At chr20:10023577 A>{G,T},
  21.66 % of tumor-half pixels differed → argmax flipped from homalt
  to homref → missing PASS.

Pending: tumor-only mode + FFPE mode (same gate per mode).

### Step 3 — Pangenome-aware DV (in progress, latest: commit `fccec22d`)

Pangenome orchestration end-to-end. Apples-to-apples (our binary vs
Docker, BOTH using the same extracted pangenome BAM as input) on
chr20:10M-10.1M:

| Run | shared | only_ours | only_docker | FM on shared |
|---|---|---|---|---|
| v1 (89 reads, no aln_*) | 252 | 60 | 70 | 9 |
| v4 (+ aln_*=2/5/10/1) | 259 | 60 | 63 | 11 |
| v5 (+ 8722-read BAM) | 259 | 39 | 63 | 2 |
| v8 (+ partition_size=25000) | 321 | 0 | 1 | 0 |
| **v9 (+ PruneLite)** | **322** | **0** | **0** | **0** |

Three flag changes closed the entire gap from 80% → 100%:

1. **v7**: Skip realigner for pangenome sample (mirrors upstream
   make_examples_core.py:2208 `can_realign`).
2. **v8**: `--partition_size=25000` matching upstream's
   run_pangenome_aware_deepvariant.py invocation. Smaller partitions
   caused the AlleleCounter's `ref_supporting_read_count` to differ
   from Docker at boundary positions.
3. **v9**: `dbg_disable_graph_pruning=true` → PruneLite (not
   min_edge_weight=0). At chr20:10035373 a long ~89bp insertion alt
   co-occurs with a C>G SNP. Our previous Prune+min_edge_weight=0
   stripped unreachable vertices, removing the alt-G haplotype path
   → reads were reassigned during realignment → no candidate
   emitted. PruneLite keeps low-weight paths, alt-G haplotype is
   preserved, candidate generated → matches Docker bit-for-bit.

Final state: 322/322 shared, 247/247 PASS, 67/67 RefCall, 8/8 NoCall,
0 GT diffs, 0 FILTER mismatches. Wall time 2 min on M4 Max
(14 threads, auto-detected). Pangenome joins WGS, DeepTrio, DeepSomatic
at 100% Docker FILTER parity on chr20:10M-10.1M.

Reference captures:

- Docker(GBZ direct)         : 327 sites (ground truth)
- Docker(our extracted BAM)  : 322 sites — 5 sites lost to BAM extraction
- Our native(BAM)            : 312 sites

Step 3-v1 (`2f65ecf2`) — orchestration end-to-end. Pangenome flags +
2-sample SampleOptions (pangenome=0, reads=1) mirroring
`make_examples_pangenome_aware_dv.py:reads_and_pangenome_samples_from_flags`.
Per-sample fields: `skip_output_generation`, `skip_phasing`,
`skip_normalization`, `keep_only_window_spanning_reads`,
`alt_aligned_pileup="none"`, `channels_enum_to_blank`. Pic-level
`sort_by_haplotypes=true`, `trim_reads_for_pileup=true`,
AlleleCounter `normalize_reads=true`. cli.cc `RunAllPangenome`
dispatch (1× make_examples + 1× call_variants + 1× postprocess).
Pangenome runs through the existing multi-sample worker (trio/somatic
sharing).

Step 3-v2 (`05e23f3a`) — `--min_mapping_quality=0` per pangenome
example_info.json:flags_for_calling. Note pangenome uses GLOBAL
default `vsc_min_fraction_{snps,indels}` (0.12 / 0.06); only mapq is
overridden.

Step 3-v3 (`18ffb771`) — `keep_legacy_allele_counter_behavior=true` +
`keep_supplementary_alignments=true` per pangenome example_info.json
(no measurable effect on chr20:10M-10.1M).

GBZ at runtime is **out of scope** for v2 (gbwt/gbwtgraph/sdsl-lite/
libdivsufsort/libhandlegraph not in Homebrew, ~5+ libs to vendor +
Boost interprocess shm). Users must convert GBZ→BAM via Docker
preprocessing once. The Docker preprocessing on chr20:10M-10.1M
produced 89 synthetic haplotype reads from `hprc-v1.1-mc-grch38.gbz`;
the BAM is reproducible via the documented pipeline (3.3 GB GBZ
download + Python script using `sam.SamReader.query`).

Pangenome model bundle: extracted via `tools/conversion/extract_weights.py`
on `/opt/models/pangenome_aware_deepvariant/wgs/` →
`pangenome.wgs.dvw` (378 tensors, 87 MB). Pangenome WGS doesn't ship
a small_model.

Probable remaining root causes for the 60 only_ours / 70 only_docker /
9 FM gap:

- **Realigner aln_* params** — we use 4/6/8/2 (match/mismatch/gap_open/
  gap_extend); pangenome wants 2/5/10/1. SSW alignment differences
  change which candidates the realigner accepts. Requires native flag
  plumbing for per-mode aln params.
- **`dbg_disable_graph_pruning=true`** — realigner's de-Bruijn graph
  pruning. Not yet wired natively; default is false.
- **GBZ→BAM extraction** caps at 322/327 ceiling (~1.5% intrinsic loss).

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
- **`std::shuffle` is implementation-defined** — libc++ (Apple Clang) and libstdc++ (GCC, Docker) produce DIFFERENT sequences for the same `mt19937_64` seed/state. This is the cause of the 1.13 % FILTER drift vs Docker on chr20: `pileup_image_native.cc::DownsampleReadIndices` shuffles read indices to subsample when coverage > 95, and our shuffle picks different reads than Docker's even when both use the same seed (2101079370). Fix: port libstdc++'s exact algorithm into `deepvariant/native/libstdcxx_shuffle.h` (paired Fisher–Yates + Lemire 128-bit uniform_int) and route `pileup_image_native.cc:162` through it. **Don't use `std::shuffle` anywhere where Docker reproducibility is required** — same applies to `std::sample`, `std::uniform_int_distribution<>` (Lemire vs rejection differs), and any other algorithm whose stdlib implementation is unspecified by the standard.
- **NumPy 1.24's `np.random.RandomState.randint` uses bitmask-rejection**, NOT Lemire. The Lemire path is in the new `Generator.integers` API. For Docker reproducibility through any `RandomState.randint(0, n)` call (used by upstream `make_examples_core.py:reservoir_sample` and elsewhere), match the legacy code path: `mask = next_pow2(n-1) - 1; do { v = next_uint32() & mask; } while (v > n - 1); return v;`. See `deepvariant/native/numpy_mt19937.h::NumpyRandomIntervalU32` and `numpy/random/src/distributions/distributions.c::random_interval` for the exact algorithm.
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
