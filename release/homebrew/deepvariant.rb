class Deepvariant < Formula
  desc "Native arm64 macOS port of Google DeepVariant (single-sample WGS)"
  homepage "https://github.com/benjamindemaille/deepvariant"
  version "0.1.0"
  license "BSD-3-Clause"
  head "https://github.com/benjamindemaille/deepvariant.git", branch: "feature/apple-silicon-native-v2"

  # Bottle-only formula: arm64 macOS only. We do NOT publish a from-source
  # build path because the CMake graph pulls in static htslib / abseil /
  # protobuf / ssw / re2 / boost and a Docker round-trip for model
  # conversion. End users get a pre-signed binary; developers build from
  # source (see README).
  bottle do
    root_url "https://github.com/benjamindemaille/deepvariant/releases/download/v#{version}"
    rebuild 0
    sha256 cellar: :any_skip_relocation, arm64_sequoia: "REPLACE_WITH_BOTTLE_SHA256"
    sha256 cellar: :any_skip_relocation, arm64_sonoma:  "REPLACE_WITH_BOTTLE_SHA256"
  end

  depends_on :macos => :sonoma           # macOS 14 floor
  depends_on arch: :arm64                # Apple Silicon only
  depends_on "deepvariant-models"        # the .mlpackage models live there

  def install
    bin.install "deepvariant"
  end

  def caveats
    <<~EOS
      DeepVariant on Apple Silicon — quick start:

        deepvariant run \\
          --reads=sample.bam \\
          --ref=ref.fa \\
          --output_vcf=out.vcf \\
          --model=#{HOMEBREW_PREFIX}/share/deepvariant-models/wgs.mlpackage \\
          --small_model_path=#{HOMEBREW_PREFIX}/share/deepvariant-models/wgs_small.mlpackage \\
          --compute_units=all

      Models live in $(brew --prefix)/share/deepvariant-models/
      (separate formula: deepvariant-models).

      For trio / somatic / pangenome, see deepvariant --help.
    EOS
  end

  test do
    assert_match "Subcommands:", shell_output("#{bin}/deepvariant 2>&1", 1)
  end
end
