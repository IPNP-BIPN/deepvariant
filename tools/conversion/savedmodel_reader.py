"""TF-free SavedModel reader.

Status: STUB. Implementation plan:

1. Vendor TF's public `.proto` files under `tools/conversion/Protos/tensorflow/`:
       saved_model.proto, meta_graph.proto, graph.proto, node_def.proto,
       attr_value.proto, tensor.proto, tensor_shape.proto, types.proto,
       op_def.proto, function.proto, versions.proto, resource_handle.proto,
       variable.proto, tensor_bundle.proto.
   Source: github.com/tensorflow/tensorflow @ r2.16 (Apache-2.0).

2. Generate Python bindings via system protoc:
       protoc --python_out=Generated/ -I=Protos/tensorflow Protos/tensorflow/...
   This needs `protobuf>=4.25` runtime; no tensorflow at all.

3. Parse `saved_model.pb` -> `SavedModel` proto -> first MetaGraphDef
   -> SignatureDef + GraphDef. Walk GraphDef.node, extract op types and
   inputs.

4. Read weights from `variables/variables.{index, data-00000-of-00001}`:
   - The index is a header (BundleHeaderProto) followed by sorted
     BundleEntryProto records keyed by variable name. We can replicate
     TF's `BundleReader` by reading the leveldb-style log format
     described at tensorflow/core/util/tensor_bundle/tensor_bundle.h.
   - For each entry: offset, length, dtype, shape -> mmap the data file
     and slice out the bytes -> numpy.ndarray with the right dtype/shape.

5. Return a dict[str, numpy.ndarray] of named tensors plus a graph
   summary that callers can use to build the equivalent torchvision /
   MLX architecture.

Until step 1 lands, every public function here raises NotImplementedError.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SavedModelReader:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not (self.directory / "saved_model.pb").exists():
            raise FileNotFoundError(f"{self.directory}/saved_model.pb not found")

    def graph_summary(self) -> dict[str, Any]:
        raise NotImplementedError(
            "SavedModelReader.graph_summary: vendor TF .proto files under Protos/tensorflow, "
            "generate Python via `protoc --python_out`, then parse saved_model.pb."
        )

    def weights(self) -> dict[str, Any]:
        raise NotImplementedError(
            "SavedModelReader.weights: implement the BundleReader equivalent "
            "(see tensorflow/core/util/tensor_bundle/tensor_bundle.{h,cc}). "
            "This needs to read variables/variables.index (leveldb-style log) "
            "and slice variables/variables.data-* by offset+length."
        )
