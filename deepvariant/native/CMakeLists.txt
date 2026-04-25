# deepvariant/native — Phase 2+: the runtime C++/Obj-C++ binary.
# Phase 1: empty placeholder — populated in Phase 2 (call_variants) and
# Phase 3 (make_examples, postprocess_variants).

# TFRecord I/O (Phase 2 dependency)
add_library(dv_tfrecord STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/tfrecord.cc"
)
target_include_directories(dv_tfrecord PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/proto_gen"
)
target_link_libraries(dv_tfrecord PUBLIC absl::crc32c proto_dv proto_nucleus)
