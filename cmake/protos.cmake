# protos.cmake — Compile protobuf schemas for DeepVariant + nucleus + TF format.
#
# Three sets of .proto files:
#   1. deepvariant/protos/           — DV variants, pileup, call output
#   2. third_party/nucleus/protos/   — reads, variants, ranges, etc.
#   3. tf_example/                   — tf.train.Example format (vendored minimal subset)
#
# Generated .pb.h/.pb.cc files land in ${CMAKE_BINARY_DIR}/proto_gen/ which
# is added to the include path of every target that links proto_dv, proto_nucleus,
# or proto_tf_example.

find_program(PROTOC protoc REQUIRED HINTS "${CMAKE_BINARY_DIR}/protobuf-src/cmake" "${CMAKE_BINARY_DIR}")

set(PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/proto_gen")
file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

# Helper macro: compile a list of .proto files and collect generated sources.
macro(dv_proto_compile TARGET PROTO_ROOT OUT_SRC_VAR)
  set(_srcs)
  foreach(_proto ${ARGN})
    cmake_path(RELATIVE_PATH _proto BASE_DIRECTORY "${PROTO_ROOT}" OUTPUT_VARIABLE _rel)
    cmake_path(REPLACE_EXTENSION _rel ".pb.cc" OUTPUT_VARIABLE _cc_rel)
    cmake_path(REPLACE_EXTENSION _rel ".pb.h"  OUTPUT_VARIABLE _hh_rel)
    set(_cc "${PROTO_GEN_DIR}/${_cc_rel}")
    set(_hh "${PROTO_GEN_DIR}/${_hh_rel}")
    add_custom_command(
      OUTPUT  "${_cc}" "${_hh}"
      COMMAND ${PROTOC}
              "--proto_path=${PROTO_ROOT}"
              "--proto_path=${CMAKE_SOURCE_DIR}/third_party/nucleus/protos"
              "--proto_path=${CMAKE_SOURCE_DIR}/third_party/tf_example"
              "--cpp_out=${PROTO_GEN_DIR}"
              "${_proto}"
      DEPENDS "${_proto}" protoc
      VERBATIM
    )
    list(APPEND _srcs "${_cc}")
  endforeach()
  set(${OUT_SRC_VAR} "${_srcs}")
endmacro()

# ---------------------------------------------------------------------------
# 1. tf.train.Example protos (vendored minimal subset — no TF runtime)
# ---------------------------------------------------------------------------
# We vendor only what call_variants / make_examples actually uses:
#   tf.train.Example, Feature, Features, BytesList, FloatList, Int64List
# The .proto files come from tools/conversion/Protos/tensorflow/, where they
# were already pulled at Phase 0. Here we expose them to the C++ build.
set(TF_EXAMPLE_PROTO_ROOT "${CMAKE_SOURCE_DIR}/third_party/tf_example")

# We don't have a separate tf_example dir yet — we'll create it as a symlink
# to the Phase 0 proto sources.
if(NOT EXISTS "${TF_EXAMPLE_PROTO_ROOT}")
  file(CREATE_LINK
    "${CMAKE_SOURCE_DIR}/tools/conversion/Protos/tensorflow"
    "${TF_EXAMPLE_PROTO_ROOT}"
    SYMBOLIC
  )
endif()

file(GLOB_RECURSE TF_EXAMPLE_PROTOS "${TF_EXAMPLE_PROTO_ROOT}/core/**/*.proto")

dv_proto_compile(proto_tf_example "${TF_EXAMPLE_PROTO_ROOT}" TF_EXAMPLE_SRCS ${TF_EXAMPLE_PROTOS})
add_library(proto_tf_example STATIC ${TF_EXAMPLE_SRCS})
target_include_directories(proto_tf_example PUBLIC "${PROTO_GEN_DIR}")
target_link_libraries(proto_tf_example PUBLIC protobuf::libprotobuf)

# ---------------------------------------------------------------------------
# 2. nucleus protos
# ---------------------------------------------------------------------------
set(NUCLEUS_PROTO_ROOT "${CMAKE_SOURCE_DIR}/third_party/nucleus/protos")
file(GLOB NUCLEUS_PROTOS "${NUCLEUS_PROTO_ROOT}/*.proto")

dv_proto_compile(proto_nucleus "${NUCLEUS_PROTO_ROOT}" NUCLEUS_SRCS ${NUCLEUS_PROTOS})
add_library(proto_nucleus STATIC ${NUCLEUS_SRCS})
target_include_directories(proto_nucleus PUBLIC "${PROTO_GEN_DIR}")
target_link_libraries(proto_nucleus PUBLIC protobuf::libprotobuf proto_tf_example)

# ---------------------------------------------------------------------------
# 3. deepvariant protos
# ---------------------------------------------------------------------------
set(DV_PROTO_ROOT "${CMAKE_SOURCE_DIR}/deepvariant/protos")
file(GLOB DV_PROTOS "${DV_PROTO_ROOT}/*.proto")

dv_proto_compile(proto_dv "${DV_PROTO_ROOT}" DV_SRCS ${DV_PROTOS})
add_library(proto_dv STATIC ${DV_SRCS})
target_include_directories(proto_dv PUBLIC "${PROTO_GEN_DIR}")
target_link_libraries(proto_dv PUBLIC protobuf::libprotobuf proto_nucleus)
