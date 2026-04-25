# deps.cmake — Fetch all external C++ dependencies (no TensorFlow).
#
# Dependencies:
#   htslib 1.18      via Homebrew (autoconf project, easiest path on macOS)
#   abseil-cpp       LTS 20240722 via FetchContent
#   protobuf 21.9    via FetchContent (matches upstream WORKSPACE pin)
#   libssw 1.2.5     via FetchContent (SSW aligner for realigner/)
#
# Pangenome deps (gbwt, gbwtgraph, sdsl-lite, libdivsufsort, libhandlegraph)
# are deferred until Phase 3 (pangenome-aware DeepVariant port).

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)  # skip re-fetch if already downloaded

# ---------------------------------------------------------------------------
# htslib 1.18 — use Homebrew on macOS (avoids autoconf complexity)
# ---------------------------------------------------------------------------
find_program(BREW_EXECUTABLE brew REQUIRED)
execute_process(
  COMMAND ${BREW_EXECUTABLE} --prefix htslib
  OUTPUT_VARIABLE HTSLIB_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT HTSLIB_PREFIX)
  message(FATAL_ERROR "htslib not found — run: brew install htslib")
endif()

# Build an IMPORTED target so we can just link htslib::htslib everywhere.
add_library(htslib::htslib STATIC IMPORTED)
find_library(HTSLIB_LIB NAMES libhts.a hts PATHS "${HTSLIB_PREFIX}/lib" REQUIRED)
set_target_properties(htslib::htslib PROPERTIES
  IMPORTED_LOCATION "${HTSLIB_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${HTSLIB_PREFIX}/include"
)
# htslib needs these system frameworks / libs on macOS.
target_link_libraries(htslib::htslib INTERFACE
  "-framework CoreFoundation"
  z bz2 lzma curl
)
message(STATUS "htslib: ${HTSLIB_LIB}")

# ---------------------------------------------------------------------------
# abseil-cpp LTS 20240722
# ---------------------------------------------------------------------------
FetchContent_Declare(
  abseil
  URL      https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.1.tar.gz
  URL_HASH SHA256=f50e5ac311a81382da7fa75b97310e4b9006474f9560ac46f54a9967f07d4ae3
)
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
FetchContent_MakeAvailable(abseil)

# ---------------------------------------------------------------------------
# protobuf 21.9  (matches upstream WORKSPACE pin)
# ---------------------------------------------------------------------------
FetchContent_Declare(
  protobuf
  URL      https://github.com/protocolbuffers/protobuf/archive/refs/tags/v21.9.zip
  URL_HASH SHA256=5babb8571f1cceafe0c18e13ddb3be556e87e12ceea3463d6b0d0064e6cc1ac3
)
set(protobuf_BUILD_TESTS OFF)
set(protobuf_BUILD_SHARED_LIBS OFF)
set(protobuf_WITH_ZLIB OFF)
# Prevent protobuf from pulling its own abseil copy — use ours.
set(protobuf_ABSL_PROVIDER "package" CACHE STRING "" FORCE)
FetchContent_MakeAvailable(protobuf)

# ---------------------------------------------------------------------------
# libssw 1.2.5 — Smith-Waterman aligner (realigner/)
# ---------------------------------------------------------------------------
FetchContent_Declare(
  libssw
  URL      https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library/archive/v1.2.5.tar.gz
  URL_HASH SHA256=b294c0cb6f0f3d578db11b4112a88b20583b9d4190b0a9cf04d83bb6a8704d9a
)
FetchContent_GetProperties(libssw)
if(NOT libssw_POPULATED)
  FetchContent_Populate(libssw)
endif()

# libssw has no CMakeLists — define targets here.
add_library(ssw STATIC
  "${libssw_SOURCE_DIR}/src/ssw.c"
  "${libssw_SOURCE_DIR}/src/ssw.h"
  "${libssw_SOURCE_DIR}/src/ssw_cpp.cpp"
  "${libssw_SOURCE_DIR}/src/ssw_cpp.h"
)
target_include_directories(ssw PUBLIC "${libssw_SOURCE_DIR}/src")
# Apple Clang/arm64: SSW uses SSE2 intrinsics guarded by __SSE2__ —
# arm64 does not have SSE2; the fallback scalar path is used automatically.
target_compile_definitions(ssw PRIVATE)

# ---------------------------------------------------------------------------
# zlib — guaranteed present on macOS (from Xcode SDK)
# ---------------------------------------------------------------------------
find_package(ZLIB REQUIRED)
