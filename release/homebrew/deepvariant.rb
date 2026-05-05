class Deepvariant < Formula
  desc "Native arm64 macOS DeepVariant — germline/trio/somatic/pangenome + Metal/ANE"
  homepage "https://github.com/benjamindemaille/deepvariant"
  version "1.10.0"
  license "BSD-3-Clause"

  # Bottle-only: arm64 macOS. Build requires htslib/abseil/protobuf/re2/boost
  # + Docker for model conversion — end users get a pre-signed binary.
  bottle do
    root_url "https://github.com/benjamindemaille/deepvariant/releases/download/v#{version}"
    rebuild 0
    sha256 cellar: :any_skip_relocation, arm64_sequoia: "REPLACE_WITH_BOTTLE_SHA256"
    sha256 cellar: :any_skip_relocation, arm64_sonoma:  "REPLACE_WITH_BOTTLE_SHA256"
  end

  depends_on :macos => :sonoma
  depends_on arch: :arm64
  depends_on "htslib"             # bgzip + tabix at runtime
  depends_on "deepvariant-models" # .mlpackage, .dvw, small-model weights, PON

  def install
    bin.install "deepvariant"
  end

  def caveats
    models = "#{HOMEBREW_PREFIX}/share/deepvariant-models"
    <<~EOS
      Quick start (models auto-discovered from deepvariant-models formula):

        # Germline WGS
        deepvariant run --reads=HG002.bam --ref=GRCh38.fa \\
          --output_vcf=out.vcf --model_type=WGS

        # DeepTrio
        deepvariant run \\
          --reads=child.bam --reads_parent1=p1.bam --reads_parent2=p2.bam \\
          --ref=ref.fa --model_type=WGS \\
          --output_vcf_child=child.vcf \\
          --output_vcf_parent1=p1.vcf --output_vcf_parent2=p2.vcf

        # DeepSomatic tumor+normal
        deepvariant somatic \\
          --reads_tumor=tumor.bam --reads_normal=normal.bam \\
          --ref=ref.fa --model_type=WGS --output_vcf=somatic.vcf

        # DeepSomatic tumor-only (with Panel-of-Normals)
        deepvariant somatic \\
          --reads_tumor=tumor.bam --ref=ref.fa \\
          --model_type=WGS_TUMOR_ONLY \\
          --population_vcfs=#{models}/deepsomatic_pon/AF_ilmn_PON_DeepVariant.GRCh38.AF0.05.vcf.gz \\
          --output_vcf=tumor_only.vcf

      ANE acceleration: add --inference_backend=ane_speculate to any command.
      Models directory: #{models}
      Override: export DEEPVARIANT_MODELS_DIR=/custom/path
    EOS
  end

  test do
    assert_match "Subcommands:", shell_output("#{bin}/deepvariant 2>&1", 1)
  end
end
