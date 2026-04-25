"""Convert a DeepVariant TF SavedModel to a Core ML .mlpackage — TF-free.

The conversion path:
  1. Read SavedModel weights via savedmodel_reader (pure protobuf, no TF).
  2. Construct torchvision.models.inception_v3 with 7-channel input.
  3. Load parsed weights into the torchvision module (manual name map).
  4. Trace the PyTorch module with an example input.
  5. coremltools.convert(traced_model, source="pytorch") -> .mlpackage.

Run inside `venv-coreml` (PyTorch + coremltools, no TF).

Usage:
    python convert_coreml.py \\
        --saved-model models/wgs --output models/wgs.mlpackage

Status: STUB. Steps 1-3 require completing savedmodel_reader.py first.
"""

from __future__ import annotations

import argparse
import sys

from savedmodel_reader import SavedModelReader


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--saved-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--input-shape", default="100,221,7")
    p.add_argument("--input-name", default="input_1")
    p.add_argument(
        "--compute-units",
        default="ALL",
        choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
    )
    p.add_argument(
        "--minimum-deployment-target",
        default="macOS14",
        choices=["macOS14", "macOS15"],
    )
    args = p.parse_args()

    reader = SavedModelReader(args.saved_model)
    try:
        graph = reader.graph_summary()
        weights = reader.weights()
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Steps 2-5 (instantiate Inception-v3 in torchvision, load weights with
    # name remapping, trace, coremltools.convert) depend on the parsed graph
    # and weight map produced above. Implementation pending.
    print(
        f"parsed graph with {len(graph.get('nodes', []))} nodes, "
        f"{len(weights)} tensors",
        file=sys.stderr,
    )
    print(
        "error: PyTorch bridge + coremltools.convert not yet implemented",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
