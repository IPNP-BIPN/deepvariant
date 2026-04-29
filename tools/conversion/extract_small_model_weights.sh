#!/usr/bin/env bash
# Extract the WGS small_model weights from the upstream Docker image
# (`google/deepvariant:1.10.0`) into a directory of FP32 NumPy `.npy`
# files. The runtime `SmallModel::Load(<dir>)` reads them and runs a
# deterministic BNNS-CPU FP32 MLP (Phase 5.5d/7) — bit-equal to
# Docker's TF/Keras inference, unlike the earlier Core ML path which
# drifted by ~0.005-0.01 on max_p and caused cross-MID FILTER flips.
#
# Output: <out_dir>/{layer_0_kernel,layer_0_bias,layer_1_kernel,
#                    layer_1_bias,layer_2_kernel,layer_2_bias}.npy
# (~2.4 MB total — 70x750 + 750x750 + 750x3 dense weights + biases).
#
# Usage:
#   tools/conversion/extract_small_model_weights.sh <out_dir>
#
# Re-run this whenever the upstream Docker image bumps the small_model
# weights. Stable across DV 1.10.0 sub-releases.

set -euo pipefail

OUT_DIR="${1:?usage: $0 <out_dir>}"
DV_VERSION="${DV_VERSION:-1.10.0}"

mkdir -p "${OUT_DIR}"
OUT_ABS="$(cd "${OUT_DIR}" && pwd)"

echo "==> Extracting small_model weights from google/deepvariant:${DV_VERSION}"
echo "    out: ${OUT_ABS}"

docker run --rm --platform linux/amd64 \
  -v "${OUT_ABS}:/out" \
  "google/deepvariant:${DV_VERSION}" \
  bash -c '
    pip install --quiet --no-warn-script-location keras 2>&1 | tail -1
    python3 -c "
import keras, numpy as np, os
m = keras.models.load_model(\"/opt/smallmodels/wgs/model.keras\", compile=False)
for i, layer in enumerate(m.layers):
    if not layer.weights:
        continue
    for w in layer.weights:
        arr = w.numpy()
        kind = \"kernel\" if \"kernel\" in w.name else (\"bias\" if \"bias\" in w.name else w.name)
        np.save(f\"/out/layer_{i}_{kind}.npy\", arr)
        print(f\"  layer_{i}_{kind}.npy  shape={arr.shape}  dtype={arr.dtype}\")
"
'

echo "==> done"
ls -la "${OUT_ABS}" | awk 'NR>1{print "    " $NF " (" $5 " B)"}'
