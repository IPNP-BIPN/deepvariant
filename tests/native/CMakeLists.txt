# Phase 1 gate tests — smoke tests that nucleus_io + realigner link correctly.
# These run under ctest: cmake --build && ctest -V must pass.
# Excludes upstream tests that depend on test_utils.h (heavy TF I/O) which
# are deferred to Phase 1b once all TF stubs are complete.

set(GATE_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/proto_gen"
  "${CMAKE_SOURCE_DIR}/cmake/tf_stubs"
  "${ABSL_PREFIX}/include"
  "${BOOST_INCLUDE_DIR}"
  "${googletest_SOURCE_DIR}/googletest/include"
  "${googletest_SOURCE_DIR}/googlemock/include"
)
set(GATE_COMPILE_OPTS
  "-include${CMAKE_SOURCE_DIR}/cmake/tf_stubs/tensorflow/core/platform/tf_compat.h"
)

function(phase1_test name src lib)
  add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
  target_include_directories(${name} PRIVATE ${GATE_INCLUDE_DIRS})
  target_compile_options(${name} PRIVATE ${GATE_COMPILE_OPTS})
  target_link_libraries(${name} PRIVATE ${lib} gtest_main gmock)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

phase1_test(nucleus_io_smoke      nucleus_io_smoke_test.cc      nucleus_io)
phase1_test(realigner_smoke       realigner_smoke_test.cc       realigner)
phase1_test(call_variants_smoke   call_variants_smoke_test.cc   dv_call_variants_lib)
phase1_test(small_model_smoke     small_model_smoke_test.cc     dv_small_model)
phase1_test(dv_weights_smoke      dv_weights_smoke_test.cc      dv_weights)
