# Architecture Decision Record — Inference Framework on Apple Silicon

**Status:** Draft (Phase 0 in progress).
**Branch:** `feature/apple-silicon-native-v2`.

## Context

DeepVariant's inference stage (`call_variants`) loads a TensorFlow SavedModel and runs Inception-v3 inference. Input shape `(N, 100, 221, 7)` for germline / `(N, 100, 221, 12)` for pangenome, output `(N, 3)` softmax. 14 stock TF ops, no custom ops.

For the Apple Silicon native port, we must pick a GPU runtime that:
- Runs natively on arm64 macOS without TensorFlow at runtime.
- Engages Metal (and ideally ANE) verifiably.
- Preserves softmax accuracy within ≤1e-3 of Linux x86 reference.
- Achieves ≥2.5× throughput vs published Linux x86 reference.
- Has a viable conversion path from the existing TF SavedModel.

Three candidates per the user's prompt:
- **Voie A:** `tensorflow-metal`.
- **Voie B:** Core ML via `coremltools`.
- **Voie C:** Apple MLX.

## Decision

_To be filled in after Phase 0 measurements (target: end of week 1)._

## Bench plan

For each candidate:

1. Convert the real `gs://deepvariant/models/DeepVariant/1.10.0/wgs/` SavedModel.
2. Bench inference at batch 1 / 8 / 32 / 128 / 1024 on M4 Max:
   - Wall-clock latency per batch.
   - Throughput (examples/sec).
   - Peak RSS.
   - GPU power & ANE power residency from `powermetrics --samplers gpu_power,ane_power -i 500` captured on a side thread.
3. Parity vs Linux x86 reference (one-time captured under Docker emulation in Phase 0):
   - Max-abs softmax difference on a 1000-example chr20 set.
   - Argmax disagreement rate (must be 0).
4. Repeat for the 12-channel pangenome model (different input shape — may force a different conversion path).

## Constraints to verify against

- **Conversion stability:** v1 documented that TF 2.20 + coremltools 9 hangs at 21 min / 3 GB during WGS conversion. v2 pins TF 2.16.2 + coremltools 7.2 for Voie B. If conversion still hangs, fall back to SavedModel→ONNX→Core ML.
- **ANE compatibility:** ANE prefers 4-ch image tensors; 7-ch / 12-ch may force GPU-only.
- **Metal determinism:** softmax tolerance (≤1e-3), not bit-equality.
- **Pangenome 12-ch path:** if Core ML rejects, this voie loses for pangenome and we may end up with a hybrid (Core ML for germline, MLX for pangenome) or fall to MLX entirely.

## Initial framework health snapshot (2026-04-25)

| Framework | Latest | Last release | TF version cap | Initial verdict |
|---|---|---|---|---|
| `tensorflow-metal` | 1.2.0 | mid-2024 | TF 2.16 only | Stale; M-series ReLU bugs reported. Risky as a production target. |
| `coremltools` | 9.0 | active | n/a (own runtime) | Solid path. macOS 14+ → ANE + GPU. v2 pins to 7.2 because of the v1 hang with 9.0. |
| `mlx` | 0.31.2 | monthly | n/a (own runtime) | Apple's strategic ML framework. Best perf on M3/M4 in published benchmarks. SavedModel ingestion via custom converter. |

## Result

_Numbers and decision pending Phase 0 execution._

| Voie | Latency b=128 (ms) | Throughput (ex/s) | GPU residency | ANE engaged | softmax max-abs | argmax disagree | Notes |
|---|---|---|---|---|---|---|---|
| A — tf-metal | TBD | TBD | TBD | n/a | TBD | TBD | |
| B — Core ML | TBD | TBD | TBD | TBD | TBD | TBD | |
| C — MLX | TBD | TBD | TBD | n/a (Metal only) | TBD | TBD | |

## Consequences (will be filled in)

_The chosen framework determines the runtime layout in Phase 2 (`deepvariant/native/coreml_inference.{h,mm}` vs `mlx_inference.{h,cpp}`). It also determines the model format shipped in `deepvariant-models` (`.mlpackage` vs MLX checkpoint)._
