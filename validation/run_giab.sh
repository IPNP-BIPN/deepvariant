#!/usr/bin/env bash
# Run our deepvariant binary on a GIAB benchmark sample and compute F1
# scores via hap.py. Targets the plan's scientific gates:
#   SNP F1   ≥ upstream F1 − 0.05 %
#   INDEL F1 ≥ upstream F1 − 0.10 %
#
# hap.py runs in Docker (it's a validation tool, not part of the binary),
# so this script needs Docker. It does NOT run our deepvariant in Docker.
#
# Usage:
#   ./validation/run_giab.sh <sample-id> [region]
#   ./validation/run_giab.sh HG002 chr20
#
# Inputs expected at $DV_GIAB_DIR (default /tmp/dv_giab/data):
#   <sample>.bam{,.bai}                — aligned reads
#   GRCh38.fa{,.fai}                   — reference
#   truth.vcf.gz{,.tbi}                — GIAB benchmark VCF
#   truth.bed                          — GIAB high-confidence regions

set -euo pipefail
cd "$(dirname "$0")/.."

SAMPLE="${1:?usage: $0 <sample> [region]}"
REGION="${2:-chr20}"
DATA="${DV_GIAB_DIR:-/tmp/dv_giab/data}"
OUT="${DV_VALIDATION_OUT:-validation/output/${SAMPLE}}"

mkdir -p "${OUT}"
echo "==> Running our deepvariant on ${SAMPLE} ${REGION}"
time ./build-macos/bin/deepvariant run \
  --reads="${DATA}/${SAMPLE}.bam" \
  --ref="${DATA}/GRCh38.fa" \
  --output_vcf="${OUT}/our.vcf" \
  --regions="${REGION}" \
  --num_shards=1 \
  --intermediate_results_dir="${OUT}/intermediate" \
  --model=tools/conversion/models/wgs.mlpackage \
  --small_model_path=tools/conversion/models/wgs_small.mlpackage \
  --compute_units=all

# Compress + index for hap.py
"${HOMEBREW_PREFIX:-/opt/homebrew}/bin/bgzip" -f "${OUT}/our.vcf"
"${HOMEBREW_PREFIX:-/opt/homebrew}/bin/tabix" -f -p vcf "${OUT}/our.vcf.gz"

echo "==> hap.py vs GIAB truth (Docker linux/amd64)"
docker run --rm --platform linux/amd64 \
  -v "${DATA}:/data:ro" \
  -v "$(realpath "${OUT}"):/work" \
  pkrusche/hap.py:latest \
  /opt/hap.py/bin/hap.py \
    /data/truth.vcf.gz \
    /work/our.vcf.gz \
    -f /data/truth.bed \
    -r /data/GRCh38.fa \
    -o /work/happy \
    --location "${REGION}"

echo
echo "==> F1 scores"
"${HOMEBREW_PREFIX:-/opt/homebrew}/bin/csvcut" -c "Type,Filter,METRIC.Recall,METRIC.Precision,METRIC.F1_Score" \
  "${OUT}/happy.summary.csv" 2>/dev/null \
  || cat "${OUT}/happy.summary.csv"
