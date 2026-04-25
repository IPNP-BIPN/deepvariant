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
