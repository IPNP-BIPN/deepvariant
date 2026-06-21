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
# Path pattern (from upstream Dockerfile, line 47-50):
#   gs://deepvariant/models/DeepVariant/<VERSION>/savedmodels/deepvariant.<NAME>.savedmodel/<file>
DV_VERSION="${DV_VERSION:-1.10.0}"
BASE="https://storage.googleapis.com/deepvariant/models/DeepVariant/${DV_VERSION}/savedmodels/deepvariant.${NAME}.savedmodel"

FILES=(
  "saved_model.pb"
  "fingerprint.pb"
  "model.example_info.json"
  "variables/variables.data-00000-of-00001"
  "variables/variables.index"
)

for f in "${FILES[@]}"; do
  url="${BASE}/${f}"
  out="${DST}/${f}"
  echo "==> fetching ${url}"
  if ! curl -fL --retry 3 --connect-timeout 15 -o "${out}" "${url}"; then
    case "${f}" in
      fingerprint.pb|model.example_info.json)
        echo "    (${f} optional, skipping)"
        rm -f "${out}"
        ;;
      *)
        echo "error: failed to fetch ${url}" >&2
        exit 1
        ;;
    esac
  fi
done

echo "==> ${NAME} SavedModel ready at ${DST}"
ls -la "${DST}" "${DST}/variables"
