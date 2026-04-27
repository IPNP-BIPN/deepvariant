# DeepVariant arm64 — Phase 4 scientific validation

GIAB hap.py F1 / precision / recall on real benchmark data. Phase 4
success gate (per the v2 plan):

- **SNP F1   ≥ upstream F1 − 0.05 %**
- **INDEL F1 ≥ upstream F1 − 0.10 %**

`upstream` here is `google/deepvariant:1.10.0` Docker run on the same
input fixture. Both pipelines are evaluated by the same hap.py tool
(`jmcdani20/hap.py:v0.3.12`) against the same GIAB v4.2.1 truth set
(`/tmp/dv_giab/data/truth.vcf.gz`) within the high-confidence regions
(`truth.bed`).

---

## Setup

- **Sample**: HG002, NIST GIAB benchmark (Genome-in-a-Bottle)
- **Reference**: GRCh38 (no decoy contigs)
- **Truth**: `truth.vcf.gz` + `truth.bed` (high-confidence regions only)
- **Aligner**: BWA-MEM 0.7.17 (BAM provided as-is, not realigned)
- **hap.py harness**: `jmcdani20/hap.py:v0.3.12` in linux/amd64 Docker
- **Hardware**: Apple M4 Max, 128 GB unified memory, macOS 26
- **Region**: HG002.chr20.bam — chr20 only (see "Scope" below)

Run by: `validation/run_giab.sh HG002 chr20`.

## Scope

The chr20 BAM exposes 19.5 M reads covering ~64.4 Mb of chr20. That's
~3 % of the human genome, but chr20 is broadly representative for
both GC content and variant density. Full-genome validation
(20-something chromosomes, ~3 GB, 2-3 hr per pipeline) is queued but
not blocking on Phase 4 gate — chr20 alone provides 71 k SNP truth
calls and 11 k INDEL truth calls, more than enough to discriminate
0.05 % F1 deltas against upstream.

## Smoke run — chr20:5M-6M (1 Mb fixture)

| Type  | Filter | TRUTH.TOTAL | TP   | FN | FP | UNK | F1     | Precision | Recall |
| ----- | ------ | ----------- | ---- | -- | -- | --- | ------ | --------- | ------ |
| INDEL | PASS   |         267 |  265 |  2 |  2 | 190 | 99.27% | 99.29%    | 99.25% |
| SNP   | PASS   |        1284 | 1272 | 12 |  0 | 108 | 99.53% | 100.00%   | 99.07% |

Wall time: 15 s (deepvariant) + 1 min (hap.py).

## chr20 full (Phase 4 gate run)

Our pipeline:

| Type  | Filter | TRUTH.TOTAL | TP    | FN  | FP | UNK   | F1     | Precision | Recall |
| ----- | ------ | ----------- | ----- | --- | -- | ----- | ------ | --------- | ------ |
| INDEL | PASS   |       11256 | 11187 |  69 | 23 |  9390 | 99.59% | 99.80%    | 99.39% |
| SNP   | PASS   |       71333 | 71008 | 325 | 45 | 16187 | 99.74% | 99.94%    | 99.54% |

Wall time:
- `deepvariant run` on chr20: **13 m 23 s** (single shard, M4 Max, ANE+GPU)
- hap.py: ~3 min

210 372 candidate variants emitted, 107 139 PASS, 80 543 RefCall,
22 690 NoCall.

### Upstream comparison

(Upstream `google/deepvariant:1.10.0` Docker run is in flight under
linux/amd64 qemu emulation; expected wall time 1-2 hr. F1 deltas
posted here once available.)

## Performance dashboard

Wall-time comparison vs upstream Docker on the same M4 Max hardware:

| Pipeline                      | chr20 wall time | speedup |
| ----------------------------- | --------------- | ------- |
| ours (native arm64, ANE+GPU)  | 13 m 23 s       | _ref_   |
| upstream Docker (qemu x86_64) | _running_       | _tbd_   |

The upstream Docker pipeline is timed under qemu emulation rather than
on a true Linux x86 host, so the throughput comparison includes a
qemu penalty — published Google reference numbers on a 64-core
EC2 c5.18xlarge are 25-40 min for full-genome WGS, so chr20 alone
should be well under that.

## Reproduction

```sh
# 1. One-time GIAB data setup (BAM, truth VCF, truth BED).
#    Layout expected at /tmp/dv_giab/data/.
ln -sf HG002.chr20.bam /tmp/dv_giab/data/HG002.bam
ln -sf HG002.chr20.bam.bai /tmp/dv_giab/data/HG002.bam.bai
tabix -p vcf -f /tmp/dv_giab/data/truth.vcf.gz

# 2. Run our pipeline + hap.py against truth.
DV_VALIDATION_OUT=validation/output/HG002_chr20 \
    ./validation/run_giab.sh HG002 chr20

# 3. F1 scores land in validation/output/HG002_chr20/happy.summary.csv
```

---

## Phase 4 gate status

- chr20 full F1 measured ✓
- Upstream Docker comparison _running_
- Full-genome validation deferred (out of scope for Phase 4 gate; would be
  Phase 4b before final release if Phase 5.5 bit-parity carries the
  per-call equivalence promise to the rest of the genome).
