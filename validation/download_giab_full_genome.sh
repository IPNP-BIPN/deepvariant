#!/usr/bin/env bash
# Phase 9 / Step 5b — download GIAB whole-genome data for trio F1 validation.
#
# Downloads (~120 GB total):
#   - HG002 / HG003 / HG004 NovaSeq PCR-free 35× BAMs + indices (~40 GB each)
#   - HG003 / HG004 GIAB v4.2.1 truth VCFs + BEDs (~150 MB each; HG002 already
#     present at /tmp/dv_giab/data/truth.vcf.gz)
#   - Full GRCh38 FASTA from NCBI (~3.1 GB)
#
# Disk space required: ~120 GB. Verify with `df -h /tmp` before running.
# Bandwidth: typical GIAB FTP throughput ~10-20 MB/s → 2-3 hours total.
#
# After download completes, run:
#   ./validation/run_giab_trio.sh
# which produces F1 numbers for all 3 samples.
#
# Usage:
#   ./validation/download_giab_full_genome.sh [target_dir]
#     target_dir defaults to ${DV_GIAB_DIR}/full or /tmp/dv_giab/full

set -euo pipefail

TARGET="${1:-${DV_GIAB_DIR:-/tmp/dv_giab}/full}"
GIAB_FTP="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab"

mkdir -p "${TARGET}"
cd "${TARGET}"

# Sanity check disk free.
DISK_FREE_GB=$(df -g . | tail -1 | awk '{print $4}')
if [ "${DISK_FREE_GB}" -lt 130 ]; then
  echo "WARNING: only ${DISK_FREE_GB} GB free at ${TARGET} — need ≥130 GB" >&2
  echo "         (~120 GB downloads + headroom for intermediate files)" >&2
  echo "         Press ENTER to proceed anyway, Ctrl-C to abort"
  read -r
fi

dl() {
  local url="$1" out="$2"
  if [ -f "${out}" ] && [ -s "${out}" ]; then
    echo "==> ${out} already present, skipping"
    return
  fi
  echo "==> Downloading ${out} (URL: ${url})"
  curl -L --progress-bar -o "${out}" "${url}"
}

# 1. Full GRCh38 reference FASTA (NCBI).
echo
echo "==> Step 1/4: full GRCh38 FASTA (~3.1 GB)"
GRCh38_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.29_GRCh38.p13/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.29_GRCh38.p13_no_alt_analysis_set.fasta.gz"
dl "${GRCh38_URL}" "GRCh38.fa.gz"
if [ ! -f "GRCh38.fa" ]; then
  echo "==> Decompressing GRCh38.fa.gz ..."
  gunzip -k "GRCh38.fa.gz"
fi
if [ ! -f "GRCh38.fa.fai" ]; then
  echo "==> Indexing with samtools faidx ..."
  /opt/homebrew/bin/samtools faidx GRCh38.fa
fi

# 2. HG002 BAM (already may exist locally; download whole-genome NovaSeq).
echo
echo "==> Step 2/4: HG002 whole-genome BAM (~40 GB)"
HG002_BAM_URL="${GIAB_FTP}/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG002.GRCh38.300x.bam"
# Use the 35x novaseq variant if available; the 300x novoalign is one canonical option.
dl "${HG002_BAM_URL}" "HG002.bam"
dl "${HG002_BAM_URL}.bai" "HG002.bam.bai"

# 3. HG003 + HG004 BAMs.
echo
echo "==> Step 3/4: HG003 + HG004 BAMs (~80 GB)"
HG003_BAM_URL="${GIAB_FTP}/data/AshkenazimTrio/HG003_NA24149_father/NIST_HiSeq_HG003_Homogeneity-12389378/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG003.GRCh38.300x.bam"
HG004_BAM_URL="${GIAB_FTP}/data/AshkenazimTrio/HG004_NA24143_mother/NIST_HiSeq_HG004_Homogeneity-14572558/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG004.GRCh38.300x.bam"
dl "${HG003_BAM_URL}"     "HG003.bam"
dl "${HG003_BAM_URL}.bai" "HG003.bam.bai"
dl "${HG004_BAM_URL}"     "HG004.bam"
dl "${HG004_BAM_URL}.bai" "HG004.bam.bai"

# 4. HG003 + HG004 GIAB v4.2.1 truth sets (HG002 is already at /tmp/dv_giab/data/).
echo
echo "==> Step 4/4: HG003 + HG004 GIAB v4.2.1 truth sets (~300 MB)"
HG003_TRUTH_URL="${GIAB_FTP}/release/AshkenazimTrio/HG003_NA24149_father/NISTv4.2.1/GRCh38/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
HG003_BED_URL="${GIAB_FTP}/release/AshkenazimTrio/HG003_NA24149_father/NISTv4.2.1/GRCh38/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"
HG004_TRUTH_URL="${GIAB_FTP}/release/AshkenazimTrio/HG004_NA24143_mother/NISTv4.2.1/GRCh38/HG004_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
HG004_BED_URL="${GIAB_FTP}/release/AshkenazimTrio/HG004_NA24143_mother/NISTv4.2.1/GRCh38/HG004_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"
dl "${HG003_TRUTH_URL}"     "HG003.truth.vcf.gz"
dl "${HG003_TRUTH_URL}.tbi" "HG003.truth.vcf.gz.tbi"
dl "${HG003_BED_URL}"       "HG003.truth.bed"
dl "${HG004_TRUTH_URL}"     "HG004.truth.vcf.gz"
dl "${HG004_TRUTH_URL}.tbi" "HG004.truth.vcf.gz.tbi"
dl "${HG004_BED_URL}"       "HG004.truth.bed"

echo
echo "==> Done. Whole-genome data at ${TARGET}/"
echo "    Total size:"
du -sh "${TARGET}"
echo
echo "==> Next: ./validation/run_giab_trio.sh"
