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
