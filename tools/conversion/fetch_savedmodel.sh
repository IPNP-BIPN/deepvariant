#!/usr/bin/env bash
# Pull a DeepVariant SavedModel from gs://deepvariant/models/DeepVariant/1.10.0/<name>/
# Usage: ./fetch_savedmodel.sh <wgs|wes|pacbio|ont_r104|hybrid_pacbio_illumina|masseq|rnaseq>
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p models

NAME="${1:?usage: $0 <model_name>}"
DST="models/${NAME}"

if [[ -f "${DST}/saved_model.pb" || -f "${DST}/saved_model.pbtxt" ]]; then
  echo "==> ${DST} already populated, skipping"
  exit 0
fi

mkdir -p "${DST}/variables"

# Public bucket; HTTPS works without credentials.
BASE="https://storage.googleapis.com/deepvariant/models/DeepVariant/1.10.0/${NAME}"

# Standard SavedModel layout. Some files are optional depending on TF version.
FILES=(
  "saved_model.pb"
  "fingerprint.pb"
  "variables/variables.data-00000-of-00001"
  "variables/variables.index"
)

for f in "${FILES[@]}"; do
  url="${BASE}/${f}"
  out="${DST}/${f}"
  echo "==> fetching ${url}"
  if ! curl -fL --retry 3 --connect-timeout 15 -o "${out}" "${url}"; then
    if [[ "${f}" == "fingerprint.pb" ]]; then
      echo "    (fingerprint.pb optional, skipping)"
      rm -f "${out}"
    else
      echo "error: failed to fetch ${url}" >&2
      exit 1
    fi
  fi
done

# Some bundles include an example_info.json or assets — try to grab them but don't fail.
for opt in "example_info.json" "assets/extra_options.json"; do
  curl -fL --retry 1 --connect-timeout 5 -o "${DST}/${opt}" \
    "${BASE}/${opt}" 2>/dev/null || true
done

echo "==> ${NAME} SavedModel ready at ${DST}"
ls -la "${DST}" "${DST}/variables"
