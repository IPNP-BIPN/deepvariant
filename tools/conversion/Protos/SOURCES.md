# Vendored protobuf sources

All `.proto` files under `Protos/tensorflow/` are vendored verbatim from upstream and **not patched**. Their license is each upstream project's own. Re-fetch via the commands below if anything changes upstream.

## TensorFlow — `tensorflow/r2.16` branch (Apache-2.0)

Source: <https://github.com/tensorflow/tensorflow>
Branch: `r2.16` (matches the TF version that DeepVariant 1.10 SavedModels were written by).
Fetched: 2026-04-25.

Files (25):

```text
core/framework/allocation_description.proto
core/framework/attr_value.proto
core/framework/cost_graph.proto
core/framework/device_attributes.proto
core/framework/full_type.proto
core/framework/function.proto
core/framework/graph.proto
core/framework/node_def.proto
core/framework/op_def.proto
core/framework/resource_handle.proto
core/framework/step_stats.proto
core/framework/tensor.proto
core/framework/tensor_description.proto
core/framework/tensor_shape.proto
core/framework/types.proto
core/framework/variable.proto
core/framework/versions.proto
core/protobuf/debug_event.proto
core/protobuf/error_codes.proto
core/protobuf/meta_graph.proto
core/protobuf/saved_model.proto
core/protobuf/saver.proto
core/protobuf/struct.proto
core/protobuf/tensor_bundle.proto
core/protobuf/trackable_object_graph.proto
```

Re-fetch:

```sh
TF_REF="r2.16"
BASE="https://raw.githubusercontent.com/tensorflow/tensorflow/${TF_REF}/tensorflow"
cd tools/conversion/Protos/tensorflow
for f in <list above>; do
  curl -fsSL -o "${f}" "${BASE}/${f}"
done
```

## Generation

Python bindings are generated under `tools/conversion/Generated/` (gitignored):

```sh
cd tools/conversion
protoc --python_out=Generated/ \
       --proto_path=Protos/tensorflow \
       Protos/tensorflow/core/**/*.proto
```

Bindings are regenerated on demand by `setup_venvs.sh` (TBD).

## Why we vendor instead of pip-install

The natural way to get TF's `.proto` definitions is `pip install tensorflow`, which we explicitly forbid (Voie B refined — TF banned in v2). Vendoring the schema files alone is ~110 KB and gives us proto bindings via `protoc --python_out` with no TF runtime.
