#!/usr/bin/env bash

set -euo pipefail

input=
output=
config=confs/tse_bsrnn_dae.yaml

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

# Dispatch preprocessing through the encoding configured for the online path.
encoding=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["dataset_args"]["cue_processing"]["textual"]["encoding"]["type"])' "${config}")
case "${encoding}" in
  dae_phoneme)
    python local/dae_kce/preprocess_phonemes.py \
      --input "${input}" --output "${output}" --config "${config}"
    ;;
  *)
    echo "Unsupported textual preprocessing: ${encoding}" >&2
    exit 1
    ;;
esac
