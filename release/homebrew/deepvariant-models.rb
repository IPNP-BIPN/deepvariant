class DeepvariantModels < Formula
  desc "Pre-converted Core ML .mlpackage models for DeepVariant on Apple Silicon"
  homepage "https://github.com/benjamindemaille/deepvariant"
  version "0.1.0"
  license "BSD-3-Clause"

  # All 25+ .mlpackage models bundled into a single archive (~3.5 GB).
  # Hosted on GitHub Releases (or R2 mirror) — fetched once on install.
  url "https://github.com/benjamindemaille/deepvariant/releases/download/v#{version}/deepvariant-models-#{version}.tar.gz"
  sha256 "REPLACE_WITH_TARBALL_SHA256"

  depends_on :macos => :sonoma
  depends_on arch: :arm64

  def install
    # Every .mlpackage and its small-model sibling lands in the share dir.
    # The deepvariant binary picks them up by --model=<path> or via the
    # DEEPVARIANT_MODELS_DIR env var.
    (share/"deepvariant-models").install Dir["*.mlpackage"]
    (share/"deepvariant-models").install Dir["*_small.mlpackage"]
  end

  def caveats
    <<~EOS
      DeepVariant models installed to:
        #{share}/deepvariant-models/

      Available variants (all FP32, bit-parity with upstream Linux x86):
        wgs, wes, pacbio, ont, hybrid, masseq, rnaseq          (DeepVariant)
        deeptrio.{wgs,wes,pacbio,ont}_{child,parent}            (DeepTrio)
        deepsomatic.{wgs,wes,pacbio,ont}{,_tumor_only}          (DeepSomatic)
        deepsomatic.{ffpe_wgs,ffpe_wes}{,_tumor_only}           (FFPE somatic)

      The deepvariant binary auto-discovers them when DEEPVARIANT_MODELS_DIR
      is unset. Override with:
        export DEEPVARIANT_MODELS_DIR=#{share}/deepvariant-models
    EOS
  end

  test do
    # Sanity: the canonical wgs model is in place.
    assert_predicate share/"deepvariant-models/wgs.mlpackage", :exist?
  end
end
