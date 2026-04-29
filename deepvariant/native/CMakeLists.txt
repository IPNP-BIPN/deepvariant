# deepvariant/native — Phase 2: call_variants (TFRecord + Core ML inference).

# ---------------------------------------------------------------------------
# dv_tfrecord — TFRecord reader/writer (C++ 17, no TF runtime)
# ---------------------------------------------------------------------------
add_library(dv_tfrecord STATIC tfrecord.cc)
target_include_directories(dv_tfrecord PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/proto_gen"
  "${ABSL_PREFIX}/include"
)
target_link_libraries(dv_tfrecord PUBLIC
  absl::crc32c
  proto_dv
  proto_nucleus
)

# ---------------------------------------------------------------------------
# dv_weights — .dvw mmap loader (Phase 5.5; consumed by Metal/BNNS path)
# ---------------------------------------------------------------------------
add_library(dv_weights STATIC dv_weights.cc)
target_include_directories(dv_weights PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${ABSL_PREFIX}/include"
)
target_link_libraries(dv_weights PUBLIC
  absl::log
)

# ---------------------------------------------------------------------------
# dv_bnns_finalize — Phase 5.5 deterministic CPU dense + softmax
# (sequential FP32 reduction, designed to bit-match TF CPU output)
# ---------------------------------------------------------------------------
add_library(dv_bnns_finalize STATIC bnns_finalize.mm)
set_source_files_properties(bnns_finalize.mm PROPERTIES
  COMPILE_FLAGS "-fobjc-arc"
)
target_include_directories(dv_bnns_finalize PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${ABSL_PREFIX}/include"
)
target_link_libraries(dv_bnns_finalize PUBLIC
  dv_weights
  absl::log
)

# ---------------------------------------------------------------------------
# dv_metal_conv_serial — Phase 5.5c deterministic-reduction-order Conv2D
# kernel (compiled at runtime via newLibraryWithSource:). Used to
# selectively replace MPSGraph convolution2D for layers that drift
# beyond the FILTER-threshold sensitivity vs Docker.
# ---------------------------------------------------------------------------
add_library(dv_metal_conv_serial STATIC metal_conv_serial.mm)
set_source_files_properties(metal_conv_serial.mm PROPERTIES
  COMPILE_FLAGS "-fobjc-arc"
)
target_include_directories(dv_metal_conv_serial PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${ABSL_PREFIX}/include"
)
target_link_libraries(dv_metal_conv_serial PUBLIC
  absl::log
  "-framework Metal"
  "-framework Foundation"
)

# ---------------------------------------------------------------------------
# dv_metal_inference — Phase 5.5 MPSGraph + BNNS Inception-v3 backend
# (Obj-C++ + Metal Performance Shaders Graph; consumes .dvw weights)
# ---------------------------------------------------------------------------
add_library(dv_metal_inference STATIC metal_inference.mm)
set_source_files_properties(metal_inference.mm PROPERTIES
  COMPILE_FLAGS "-fobjc-arc"
)
target_include_directories(dv_metal_inference PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${ABSL_PREFIX}/include"
)
target_link_libraries(dv_metal_inference PUBLIC
  dv_weights
  dv_metal_conv_serial
  absl::log
  "-framework Metal"
  "-framework MetalPerformanceShadersGraph"
  "-framework Foundation"
)

# ---------------------------------------------------------------------------
# dv_coreml — Obj-C++ Core ML inference wrapper
# (Obj-C++ requires macOS frameworks; no Python, no TF)
# ---------------------------------------------------------------------------
add_library(dv_coreml STATIC coreml_inference.mm)
set_source_files_properties(coreml_inference.mm PROPERTIES
  COMPILE_FLAGS "-fobjc-arc"
)
target_include_directories(dv_coreml PUBLIC
  "${CMAKE_SOURCE_DIR}"
)
target_link_libraries(dv_coreml PUBLIC
  "-framework CoreML"
  "-framework Foundation"
)

# ---------------------------------------------------------------------------
# dv_small_model — deterministic FP32 BNNS-CPU MLP runner for the
# small_model (70 → 750 → 750 → 3). Reads weights from .npy files
# (Phase 5.5d/7); replaces the Core ML path which had ~0.005-0.01 drift
# vs Docker's TF/Keras output and caused cross-MID FILTER flips at
# threshold-borderline sites.
# ---------------------------------------------------------------------------
add_library(dv_small_model STATIC
  small_model_inference.mm
  small_model_features.cc
)
target_include_directories(dv_small_model PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/proto_gen"
  "${ABSL_PREFIX}/include"
)
target_link_libraries(dv_small_model PUBLIC
  proto_dv
  proto_nucleus
  absl::log
)

# ---------------------------------------------------------------------------
# dv_call_variants_lib — the call_variants logic (C++, no Obj-C)
# ---------------------------------------------------------------------------
add_library(dv_call_variants_lib STATIC call_variants_main.cc)
target_include_directories(dv_call_variants_lib PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/proto_gen"
  "${ABSL_PREFIX}/include"
  "${CMAKE_SOURCE_DIR}/cmake/tf_stubs"
)
target_compile_options(dv_call_variants_lib PRIVATE
  "-include${CMAKE_SOURCE_DIR}/cmake/tf_stubs/tensorflow/core/platform/tf_compat.h"
)
target_link_libraries(dv_call_variants_lib PUBLIC
  dv_tfrecord
  dv_coreml
  dv_metal_inference
  dv_bnns_finalize
  proto_dv
  proto_nucleus
  absl::flags
  absl::flags_parse
  absl::log
  absl::strings
)

# ---------------------------------------------------------------------------
# Phase 2 smoke test
# (also in tests/native/ — added later after this lib compiles)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 3 — TF-free replacement for nucleus::ExampleWriter
# (originally implemented against tensorflow::io::RecordWriter)
# ---------------------------------------------------------------------------
add_library(dv_example_writer STATIC
  "${CMAKE_SOURCE_DIR}/patches/example_writer_macos.cc"
)
target_include_directories(dv_example_writer PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/proto_gen"
  "${CMAKE_SOURCE_DIR}/cmake/tf_stubs"
  "${ABSL_PREFIX}/include"
)
target_compile_options(dv_example_writer PRIVATE
  "-include${CMAKE_SOURCE_DIR}/cmake/tf_stubs/tensorflow/core/platform/tf_compat.h"
)
target_link_libraries(dv_example_writer PUBLIC
  dv_tfrecord
  proto_nucleus
  absl::log
  absl::strings
  absl::status
)

# ---------------------------------------------------------------------------
# Phase 3 — make_examples orchestration
# ---------------------------------------------------------------------------
set(ME_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/proto_gen"
  "${CMAKE_SOURCE_DIR}/cmake/tf_stubs"
  "${ABSL_PREFIX}/include"
  "${BOOST_INCLUDE_DIR}"
)
set(ME_COMPILE_OPTS
  "-include${CMAKE_SOURCE_DIR}/cmake/tf_stubs/tensorflow/core/platform/tf_compat.h"
)

add_library(dv_make_examples_lib STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/make_examples_main.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/regions.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/realigner_native.cc"
)
target_include_directories(dv_make_examples_lib PUBLIC ${ME_INCLUDE_DIRS})
target_compile_options(dv_make_examples_lib PRIVATE ${ME_COMPILE_OPTS})
target_link_libraries(dv_make_examples_lib PUBLIC
  dv_make_examples_native
  dv_allelecounter
  dv_tfrecord
  dv_small_model
  realigner
  proto_dv
  proto_nucleus
  absl::flags
  absl::flags_parse
  absl::log
  absl::strings
  nucleus_io
)

# ---------------------------------------------------------------------------
# Phase 3 — postprocess_variants orchestration
# ---------------------------------------------------------------------------
add_library(dv_postprocess_lib STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/postprocess_main.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/haplotypes.cc"
)
target_include_directories(dv_postprocess_lib PUBLIC ${ME_INCLUDE_DIRS})
target_compile_options(dv_postprocess_lib PRIVATE ${ME_COMPILE_OPTS})
target_link_libraries(dv_postprocess_lib PUBLIC
  dv_tfrecord
  proto_dv
  proto_nucleus
  absl::flags
  absl::flags_parse
  absl::log
  absl::strings
  nucleus_io
)

# ---------------------------------------------------------------------------
# debug_metal — dev-only diagnostic for Phase 5.5 MPSGraph divergence.
# Compares Metal stem_s1a output to the hand-computed reference for an
# all-zeros input.
# ---------------------------------------------------------------------------
add_executable(debug_metal "${CMAKE_CURRENT_SOURCE_DIR}/debug_metal_main.cc")
target_include_directories(debug_metal PRIVATE ${ME_INCLUDE_DIRS})
target_compile_options(debug_metal PRIVATE ${ME_COMPILE_OPTS})
target_link_libraries(debug_metal PRIVATE
  dv_metal_inference
  dv_weights
)

# ---------------------------------------------------------------------------
# microtest_numpy_rng — Phase 5.5d/3 verification that NumpyMt19937 +
# BoundedLemireUint32 reproduce NumPy 1.24's
# np.random.RandomState(seed).randint(...) bit-for-bit.
# ---------------------------------------------------------------------------
add_executable(microtest_numpy_rng
  "${CMAKE_CURRENT_SOURCE_DIR}/microtest_numpy_rng.cc"
)
target_include_directories(microtest_numpy_rng PRIVATE
  "${CMAKE_SOURCE_DIR}"
)

# ---------------------------------------------------------------------------
# microtest_conv_serial — Phase 5.5c hand-verifiable test for the
# deterministic Conv2D kernel. Compares GPU dispatch output against a
# scalar (kh,kw,c_in)-order CPU reference using std::fma — bit-exact
# match expected on healthy build.
# ---------------------------------------------------------------------------
add_executable(microtest_conv_serial
  "${CMAKE_CURRENT_SOURCE_DIR}/microtest_conv_serial.mm"
)
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/microtest_conv_serial.mm" PROPERTIES
  COMPILE_FLAGS "-fobjc-arc"
)
target_include_directories(microtest_conv_serial PRIVATE
  ${ME_INCLUDE_DIRS}
  "${CMAKE_SOURCE_DIR}"
  "${ABSL_PREFIX}/include"
)
target_link_libraries(microtest_conv_serial PRIVATE
  dv_metal_conv_serial
  absl::log
  "-framework Metal"
  "-framework Foundation"
)

# ---------------------------------------------------------------------------
# extract_pileup_npy — dev-only profiling tool for Phase 5.5c per-layer
# drift work. Reads N pileup images from a TFRecord (or `name@N` shard
# spec) and writes a NumPy `(N, 100, 221, 7)` FP32 NHWC array. The
# resulting `.npy` is consumed by `dump_tf_per_layer.py` (Docker) and
# `debug_metal --compare-to-reference` to profile drift on real data.
# ---------------------------------------------------------------------------
add_executable(extract_pileup_npy
  "${CMAKE_CURRENT_SOURCE_DIR}/extract_pileup_npy_main.cc"
)
target_include_directories(extract_pileup_npy PRIVATE ${ME_INCLUDE_DIRS})
target_link_libraries(extract_pileup_npy PRIVATE dv_tfrecord)

# extract_pileup_at_pos — locate a specific (chrom, start, ref, alt)
# in an examples.tfrecord and dump that single pileup as a (1,100,221,7)
# NHWC FP32 .npy. Used to byte-compare our pileup vs Docker's at a
# PASS-flip site (Phase 5.5c diagnostic).
add_executable(extract_pileup_at_pos
  "${CMAKE_CURRENT_SOURCE_DIR}/extract_pileup_at_pos_main.cc"
)
target_include_directories(extract_pileup_at_pos PRIVATE ${ME_INCLUDE_DIRS})
target_link_libraries(extract_pileup_at_pos PRIVATE dv_tfrecord)

# ---------------------------------------------------------------------------
# microtest_metal — hand-verifiable MPSGraph conv micro-tests for
# Phase 5.5a investigation. Builds tiny graphs (1×1 and 3×3) with
# inputs / weights small enough to compute the expected output by
# pencil-and-paper, and prints PASS/FAIL per test.
# ---------------------------------------------------------------------------
add_executable(microtest_metal
  "${CMAKE_CURRENT_SOURCE_DIR}/microtest_main.mm"
)
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/microtest_main.mm" PROPERTIES
  COMPILE_FLAGS "-fobjc-arc"
)
target_include_directories(microtest_metal PRIVATE ${ME_INCLUDE_DIRS})
target_link_libraries(microtest_metal PRIVATE
  "-framework Metal"
  "-framework MetalPerformanceShadersGraph"
  "-framework Foundation"
)

# ---------------------------------------------------------------------------
# dump_cvo — dev-only TFRecord dumper for CallVariantsOutput protos.
# Used to diff our candidate set against upstream's during parity work.
# ---------------------------------------------------------------------------
add_executable(dump_cvo
  "${CMAKE_CURRENT_SOURCE_DIR}/dump_cvo_main.cc"
)
target_include_directories(dump_cvo PRIVATE ${ME_INCLUDE_DIRS})
target_compile_options(dump_cvo PRIVATE ${ME_COMPILE_OPTS})
target_link_libraries(dump_cvo PRIVATE
  dv_tfrecord
  proto_dv
  proto_nucleus
  absl::log
  absl::strings
  absl::hash
)

# ---------------------------------------------------------------------------
# dump_allele_counts — dev-only AlleleCount dumper.
# Useful for diffing per-position ref/alt counts between our pipeline
# and upstream during candidate-set parity work.
# ---------------------------------------------------------------------------
add_executable(dump_allele_counts
  "${CMAKE_CURRENT_SOURCE_DIR}/dump_allele_counts_main.cc"
)
target_include_directories(dump_allele_counts PRIVATE ${ME_INCLUDE_DIRS})
target_compile_options(dump_allele_counts PRIVATE ${ME_COMPILE_OPTS})
target_link_libraries(dump_allele_counts PRIVATE
  dv_allelecounter
  proto_dv
  proto_nucleus
  nucleus_io
  absl::log
  absl::strings
  absl::hash
  absl::flat_hash_map
  absl::flat_hash_set
  absl::raw_hash_set
)

# ---------------------------------------------------------------------------
# deepvariant — main binary (CLI dispatcher)
# ---------------------------------------------------------------------------
add_executable(deepvariant
  "${CMAKE_CURRENT_SOURCE_DIR}/cli.cc"
)
target_include_directories(deepvariant PRIVATE ${ME_INCLUDE_DIRS})
target_compile_options(deepvariant PRIVATE ${ME_COMPILE_OPTS})
target_link_libraries(deepvariant PRIVATE
  dv_make_examples_lib
  dv_postprocess_lib
  dv_call_variants_lib
  dv_example_writer
  absl::flags
  absl::flags_parse
  absl::log
  absl::log_initialize
)
