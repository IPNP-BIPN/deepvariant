# Phase 0 — Inference framework bench (dev-time Python tooling)

> **Dev-time only — never shipped to users.** Python lives here because `coremltools`, `tf2onnx`, `tensorflow-metal`, and the SavedModel reader for MLX are all Python-only Apple/Google packages. Re-implementing them in Swift would re-introduce 7+ years of edge-case bug fixes (BatchNorm fused vs not, ConcatV2 axis handling, NHWC↔NCHW layout, FusedBatchNormV3 epsilon, etc.). Voie B accepts dev-time Python to keep this maturity, while the user-facing `deepvariant` binary stays 100 % C++/Obj-C++ with no embedded Python interpreter (verified by `otool -L` in Phase 5). See `CLAUDE.md` for the dev-time vs runtime split.

Three-way A/B/C bench (tensorflow-metal vs Core ML vs MLX) on the real WGS SavedModel: produces latency, throughput, GPU/ANE residency, and parity-vs-Linux measurements that feed the Phase 0 ADR.

## Layout

```text
tools/conversion/
├── .python-version            # pyenv pin: 3.11.x
├── requirements-coreml.txt    # TF 2.16.2 + coremltools 7.2
├── requirements-metal.txt     # TF 2.16.2 + tensorflow-metal 1.2.0
├── requirements-mlx.txt       # MLX 0.21+ (TF only for SavedModel weight ingest)
├── setup_venvs.sh             # creates venv-coreml / venv-metal / venv-mlx
├── fetch_savedmodel.sh        # pulls gs://deepvariant/models/DeepVariant/1.10.0/<name>
├── convert_coreml.py          # SavedModel -> .mlpackage (coremltools, with tf2onnx fallback)
├── convert_metal.py           # passthrough (records TF/metal env metadata)
├── convert_mlx.py             # SavedModel weights -> safetensors (MLX-friendly)
├── bench.py                   # latency + powermetrics GPU residency + softmax capture
├── parity_check.py            # softmax max-abs / argmax disagreement vs reference
└── models/                    # gitignored, where SavedModels and outputs live
```

## Pipeline

```sh
# one-time setup (~30 min: pyenv install 3.11 + 3 venv pip installs)
./setup_venvs.sh

# pull the real WGS SavedModel (~700 MB)
./fetch_savedmodel.sh wgs

# convert each voie
source venv-coreml/bin/activate
python convert_coreml.py --saved-model models/wgs --output models/wgs.mlpackage
deactivate

source venv-metal/bin/activate
python convert_metal.py --saved-model models/wgs --output models/wgs.metal.json
deactivate

source venv-mlx/bin/activate
python convert_mlx.py --saved-model models/wgs --output models/wgs.mlx.safetensors
deactivate

# bench each on the same 1000-example reference set
source venv-coreml/bin/activate
python bench.py --backend coreml --model models/wgs.mlpackage \
  --examples ../reference/cache/wgs_chr20_1000.tfrecord \
  --output ../../benchmarks/coreml_wgs.json \
  --output-cv ../../benchmarks/coreml_wgs.cv.tfrecord
deactivate
# (repeat for metal, mlx)

# parity vs Linux reference
python parity_check.py \
  --reference ../reference/output/wgs/call_variants_chr20.tfrecord \
  --candidates ../../benchmarks/coreml_wgs.cv.tfrecord \
              ../../benchmarks/metal_wgs.cv.tfrecord \
              ../../benchmarks/mlx_wgs.cv.tfrecord
```

## Why three pinned venvs

These three frameworks have **incompatible** Python/TF/numpy version requirements:

| venv | Python | TF | Other |
| --- | --- | --- | --- |
| venv-coreml | 3.11 | 2.16.2 | coremltools 7.2 (v1 found 9.0 + TF 2.20 hangs at 21 min / 3 GB RSS) |
| venv-metal | 3.11 | 2.16.2 | tensorflow-metal 1.2.0 (frozen at TF 2.16 since mid-2024) |
| venv-mlx | 3.11 | 2.16.2 (read-only for SavedModel ingest) | MLX 0.21+ |

We pin `numpy < 2` everywhere because TF 2.16 was built against numpy 1.

## Stop conditions

- If `convert_coreml.py` hangs > 5 min on the WGS SavedModel, bail and try `convert_coreml.py --via-onnx` (fallback path: SavedModel → ONNX → Core ML via tf2onnx).
- If the parity check shows argmax disagreement on **any** of the 1000 examples, the framework is rejected — no exceptions.
- If `bench.py` reports `gpu_power=0` AND `ane_power=0` for a backend, that backend has a config bug, not a perf result.
