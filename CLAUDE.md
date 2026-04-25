# CLAUDE.md — DeepVariant Apple Silicon Native Port (v2)

Project memory for AI-assisted work on `feature/apple-silicon-native-v2`.

## What this branch is

A fresh-start port of Google DeepVariant (and DeepTrio, DeepSomatic, pangenome-aware DV) to a single, fully native arm64 binary on Apple Silicon, distributed via Homebrew, with Apple Metal GPU + ANE inference and **zero Python interpreter at runtime**.

Authoritative plan: `~/.claude/plans/prompt-deepvariant-apple-idempotent-peacock.md`.
Running log: `PORT_LOG.md`.

## Hard constraints (non-negotiable)

- macOS ≥ 14, arm64 only.
- No Docker / no Rosetta / no CUDA / no embedded Python interpreter shipped to the user.
- Build is reproducible. User installs in one Homebrew command, no compilation on their box.
- **Scientific accuracy preserved**: SNP F1 ≥ reference − 0.05 %, INDEL F1 ≥ reference − 0.10 %.
- **GPU truly engaged**: verified by `powermetrics --samplers gpu_power,ane_power` showing non-zero residency.
- **Speedup ≥ 2.5×** vs published Linux x86 reference.

## Working rules

1. **Test before commit.** Every commit must leave the build green: `cmake --build build && ctest -V` (after Phase 1) or, for Phase 0, the conversion + parity-check scripts must run end-to-end.
2. **Never degrade scientific precision.** F1 thresholds are gates, not goals. If we slip below, we fix the root cause — we do not lower the bar.
3. **Never bypass an error.** No `--no-verify`, no swallowed exceptions, no commenting out of failing tests. Diagnose the root cause.
4. **Document every critical decision** in `PORT_LOG.md` with date, context, alternatives considered, and rationale.
5. **Don't touch the v1 worktree** at `/Users/benjamin/projects/deepvariant-apple-silicon/.worktrees/apple-silicon-native/`. v1 is a separate clone retained as research; v2 is its own fresh history.
6. **Don't modify upstream `BUILD` / Bazel rules.** They stay as a Linux/Bazel reference. v2 builds via CMake on macOS only.
7. **No half-finished implementations.** Each phase has a success gate; do not cross it without meeting the gate.

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

## Pitfalls already known (mine before re-discovering)

- **TF 2.20 + coremltools 9 hangs** during WGS SavedModel conversion (21 min / 3 GB RSS). Pin TF 2.16.x in the conversion venv.
- **tensorflow-metal 1.2.0** is frozen at TF 2.16; M-series ReLU bugs reported. Include in bench, expect to lose.
- **ANE prefers 4-channel image-shaped tensors.** Our model is 7- or 12-channel. ANE may refuse — accept GPU-only fallback.
- **Metal compute is not bitwise reproducible** across some ops/reboots. Validate via softmax tolerance (≤1e-3) + argmax agreement, not bit-equality.
- **`build-prereq.sh` is Linux-only.** v2 ships `scripts/build-prereq-macos.sh`.
- **8.5 GB of model artifacts** can't fit in a single Homebrew bottle alongside the binary. Split into `deepvariant-models` formula.
- **Xcode CLT is enough — no full Xcode required.** Ship `.mlpackage` uncompiled; runtime compiles on first load via `MLModel compileModelAtURL:error:`. Avoid `xcrun coremlcompiler` (full Xcode only).

## Key file paths

- Plan: `~/.claude/plans/prompt-deepvariant-apple-idempotent-peacock.md`
- v2 root: `/Users/benjamin/deepvariant`
- v1 reference clone: `/Users/benjamin/projects/deepvariant-apple-silicon/.worktrees/apple-silicon-native/` (read-only)
- Native runtime (Phases 2-3): `deepvariant/native/`
- Build (Phase 1): `CMakeLists.txt` + `cmake/*.cmake`
- Conversion (Phase 0, dev-time): `tools/conversion/`
- Linux ref capture (Phase 0): `tools/reference/`
- Release tooling (Phase 5): `release/`
- Homebrew formulas (Phase 6): separate repo `homebrew-deepvariant/`

## Reused upstream C++ (do not rewrite)

These are the multipliers that make v2 feasible. Wrap, don't rewrite:

- `deepvariant/make_examples_native.cc`
- `deepvariant/pileup_image_native.cc`
- `deepvariant/allelecounter.cc`
- `deepvariant/realigner/{fast_pass_aligner,debruijn_graph,ssw,window_selector}.cc`
- `deepvariant/{direct_phasing,merge_variants,merge_phased_reads,postprocess_variants}.cc`
- `third_party/nucleus/io/{sam_reader,vcf_reader,vcf_writer,reference,gbz_reader}.cc`
