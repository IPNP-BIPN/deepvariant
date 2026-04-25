# Vendored TF `.proto` files (TF-free SavedModel reading)

(Empty in this scaffold commit.)

To enable the TF-free SavedModel reader in `savedmodel_reader.py`, vendor these public protobuf schemas verbatim from `github.com/tensorflow/tensorflow` at `r2.16` (Apache-2.0):

```
Protos/tensorflow/core/protobuf/saved_model.proto
Protos/tensorflow/core/protobuf/meta_graph.proto
Protos/tensorflow/core/framework/graph.proto
Protos/tensorflow/core/framework/node_def.proto
Protos/tensorflow/core/framework/attr_value.proto
Protos/tensorflow/core/framework/tensor.proto
Protos/tensorflow/core/framework/tensor_shape.proto
Protos/tensorflow/core/framework/types.proto
Protos/tensorflow/core/framework/op_def.proto
Protos/tensorflow/core/framework/function.proto
Protos/tensorflow/core/framework/versions.proto
Protos/tensorflow/core/framework/resource_handle.proto
Protos/tensorflow/core/framework/variable.proto
Protos/tensorflow/core/util/tensor_bundle/tensor_bundle.proto
```

Then generate Python bindings (no TF runtime):

```sh
brew install protobuf
protoc --python_out=Generated/ -I=Protos/tensorflow Protos/tensorflow/...
```

`SOURCES.md` (next to this file) records the exact upstream commit hash for each vendored file.
