# DeepVariant Apple Silicon Native Port — v2 PORT_LOG

Running log of decisions, gotchas, and progress on `feature/apple-silicon-native-v2`.

Plan reference: `~/.claude/plans/prompt-deepvariant-apple-idempotent-peacock.md`.

## 2026-04-25 — Phase 0 bootstrap

Branch `feature/apple-silicon-native-v2` created from `origin/r1.10` at commit `45f26275`.

Scaffolding directories created:
- `patches/` — local patches against vendored deps and upstream sources.
- `benchmarks/` — Phase 0 latency / GPU residency captures.
- `packaging/` — release artifacts and bottle staging.
- `tools/conversion/` — dev-time TF→{Core ML, MLX, tf-metal} converter scripts.
- `tools/reference/` — one-time Linux x86 reference capture under Docker emulation.
- `release/` — sign, notarize, model-conversion CI scripts.
- `cmake/` — CMake module files (Phase 1).
- `deepvariant/native/` — new pure-C++/Obj-C++ runtime (Phases 2-3).
- `validation/` — GIAB hap.py harness (Phase 4) and virgin-machine checklist (Phase 7).

### System snapshot

| Item | Value |
|---|---|
| Date | 2026-04-25T22:49:07+0200 |
| OS | macOS 26.4.1 (build 25E253) |
| Arch | arm64 |
| CPU | Apple M4 Max |
| RAM | 128 GB unified |
| Xcode | **CLT only** (`/Library/Developer/CommandLineTools`) — sufficient (see decision below). |
| Apple Clang | 21.0.0 (clang-2100.0.123.102) |
| CMake | 4.3.2 (Homebrew) |
| protoc (system) | libprotoc 34.1 — informational only; we vendor protobuf 21.9 statically per plan |
| Python (system) | 3.12.13 |
| pyenv | 2.6.27 — used for the three Phase 0 conversion venvs |
| Docker | 29.2.1 — used **dev-time only** to capture Linux x86 reference under qemu emulation; never shipped |
| Homebrew | 5.1.7 (`/opt/homebrew/bin/brew`) |

### Notes from prior v1 attempt

A previous v1 worktree exists at `/Users/benjamin/projects/deepvariant-apple-silicon/.worktrees/apple-silicon-native/` (separate clone, not a worktree of this repo). v1 reached a Phase 0 ADR favoring Core ML and bumped Bazel/Python/TF toolchain pins. v2 is a fresh start per the user's choice. v1 findings retained for reference only:
- TF 2.20 + coremltools 9 hangs at 21 min / 3 GB RSS during real WGS SavedModel→Core ML conversion. **v2 will pin TF 2.16.x in the conversion venv.**
- `tensorflow-metal` is frozen at TF 2.16 since mid-2024 and reports M-series ReLU bugs. v2 includes it in the bench for completeness but expectation is Core ML or MLX wins.
- `make_examples_native.cc`, `pileup_image_native.cc`, `allelecounter.cc`, the realigner C++, and `direct_phasing.cc` are all reusable — they form the multipliers that make v2 feasible.

### Build system: Bazel → CMake (decided)

v2 abandons Bazel for the native build. Upstream's Bazel rules transitively require `@org_tensorflow`, which we do not want at runtime. CMake is ~equivalent effort and produces a self-contained TF-free graph. Upstream `BUILD` files are left untouched as a Linux/Bazel reference for cross-checking.

### Xcode CLT only — no full Xcode needed (decided)

The plan flagged full Xcode as a possible Phase 5 requirement (for `xcrun coremlcompiler`). Re-evaluated: not needed.

- `coremlcompiler` is bundled with the full Xcode app (`Xcode.app/Contents/Developer/usr/bin/coremlcompiler`) and pre-compiles `.mlpackage` → `.mlmodelc`.
- Alternative: ship `.mlpackage` uncompiled in `deepvariant-models`; the binary calls `[MLModel compileModelAtURL:url error:&err]` at first load. Result is cached by Core ML in `~/Library/Caches/com.apple.CoreML/`. No Xcode needed anywhere.
- Cost: first run after install adds ~few seconds per model used while Core ML compiles. Subsequent runs are unaffected. We log a clear `Compiling Core ML model for first run…` line.
- Everything else (`clang`, `codesign`, `xcrun notarytool`, `xcrun stapler`, `MacOSX.sdk` with `CoreML.framework` headers) ships with CLT.

This keeps the build/release machine on CLT only, which is also more reproducible (CLT versions are easier to pin than Xcode versions).

### Next milestone — Phase 0 step 1

Create three pinned conversion venvs in `tools/conversion/`:
- `venv-coreml` (Python 3.11, TF 2.16.2, coremltools 7.2)
- `venv-metal` (Python 3.11, TF 2.16.2, tensorflow-metal 1.2)
- `venv-mlx` (Python 3.11, MLX 0.21+)

Pull `gs://deepvariant/models/DeepVariant/1.10.0/wgs/` SavedModel into a pinned cache.

Then proceed with `convert_coreml.py`, `convert_metal.py`, `convert_mlx.py`.
