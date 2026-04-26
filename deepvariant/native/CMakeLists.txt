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
# dv_small_model — Obj-C++ wrapper around the small_model .mlpackage
# ---------------------------------------------------------------------------
add_library(dv_small_model STATIC
  small_model_inference.mm
  small_model_features.cc
)
set_source_files_properties(small_model_inference.mm PROPERTIES
  COMPILE_FLAGS "-fobjc-arc"
)
target_include_directories(dv_small_model PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/proto_gen"
  "${ABSL_PREFIX}/include"
)
target_link_libraries(dv_small_model PUBLIC
  proto_dv
  proto_nucleus
  "-framework CoreML"
  "-framework Foundation"
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
)
target_include_directories(dv_make_examples_lib PUBLIC ${ME_INCLUDE_DIRS})
target_compile_options(dv_make_examples_lib PRIVATE ${ME_COMPILE_OPTS})
target_link_libraries(dv_make_examples_lib PUBLIC
  dv_make_examples_native
  dv_allelecounter
  dv_tfrecord
  dv_small_model
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
