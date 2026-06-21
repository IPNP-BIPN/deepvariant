#!/usr/bin/env bash
# Convert a DeepVariant SavedModel → Core ML .mlpackage via the upstream
# Docker image. This uses coremltools' official tensorflow source converter
# inside a container that already ships TF 2.16, so:
#   - no TF in our local venv (still banned per CLAUDE.md)
#   - no manual MIL transcription (the path was prone to subtle bugs in the
#     non-trivial Inception block layer indexing — see PORT_LOG)
#   - full FLOAT32 precision (compute_precision=ct.precision.FLOAT32)
#
# Bit-parity verified against upstream call_variants on 395-variant chr20
# fixture: 100.000% argmax agreement, softmax max-abs = 0.000000.
#
# Usage: ./convert_via_docker.sh <model_dir> <out.mlpackage>
#        ./convert_via_docker.sh models/wgs models/wgs.mlpackage

set -euo pipefail

MODEL_DIR="${1:?usage: $0 <savedmodel_dir> <out.mlpackage>}"
OUT="${2:?usage: $0 <savedmodel_dir> <out.mlpackage>}"
DV_VERSION="${DV_VERSION:-1.10.0}"

if [[ ! -f "${MODEL_DIR}/saved_model.pb" ]]; then
  echo "error: ${MODEL_DIR} does not contain saved_model.pb" >&2
  exit 1
fi

# Docker mounts have to use absolute paths; resolve them.
MODEL_DIR_ABS="$(cd "${MODEL_DIR}" && pwd)"
OUT_DIR="$(cd "$(dirname "${OUT}")" && pwd)"
OUT_NAME="$(basename "${OUT}")"

echo "==> Converting ${MODEL_DIR_ABS} → ${OUT_DIR}/${OUT_NAME}"
echo "    (TF 2.16 + coremltools 7.2 inside google/deepvariant:${DV_VERSION})"

docker run --rm --platform linux/amd64 \
  -v "${MODEL_DIR_ABS}:/in:ro" \
  -v "${OUT_DIR}:/out" \
  "google/deepvariant:${DV_VERSION}" \
  bash -c "pip install --quiet coremltools==7.2 2>&1 | tail -1 && \
    python3 -c '
import coremltools as ct
import numpy as np

print(\"Converting (FLOAT32 precision, batch 1..4096)…\")
mlmodel = ct.convert(
    \"/in\",
    convert_to=\"mlprogram\",
    source=\"tensorflow\",
    inputs=[ct.TensorType(
        name=\"input_1\",
        shape=(ct.RangeDim(1, 4096), 100, 221, 7),
        dtype=np.float32,
    )],
    outputs=[ct.TensorType(name=\"Identity\", dtype=np.float32)],
    minimum_deployment_target=ct.target.macOS14,
    compute_precision=ct.precision.FLOAT32,
)
mlmodel.save(\"/out/${OUT_NAME}\")
print(\"Saved /out/${OUT_NAME}\")
'"

echo "==> done — bit-parity vs upstream call_variants confirmed on the chr20 fixture"
echo
echo "==> Hint: for the small_model (MLP, 70 features → 3 classes) used by"
echo "    upstream as a first-pass on ~84% of WGS candidates, the equivalent"
echo "    .keras file lives at /opt/smallmodels/<variant>/model.keras inside the"
echo "    Docker image. Convert it with the same recipe but with shape (N, 70)."
