#!/usr/bin/env bash

set -euo pipefail

if [ -z "${WESEP_ROOT:-}" ]; then
  . "$(dirname "${BASH_SOURCE[0]}")/../path.sh"
fi

mix_data_path=
librispeech_root=
data=data
noise_type=clean
textual_input=text
config=confs/tse_bsrnn_dae.yaml

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${mix_data_path}" ] || [ -z "${librispeech_root}" ]; then
  echo "--mix-data-path and --librispeech-root are required." >&2
  exit 1
fi
if [ ! -d "${mix_data_path}" ] || [ ! -d "${librispeech_root}" ]; then
  echo "Libri2Mix or LibriSpeech data root does not exist." >&2
  exit 1
fi
case "${textual_input}" in
  text) cue_format=word_text; cue_index=textual.json ;;
  dae_phoneme) cue_format=dae_keyword_phoneme; cue_index=textual_phoneme.json ;;
  *) echo "Unsupported textual input: ${textual_input}" >&2; exit 1 ;;
esac

# Cue YAML files differ only in their resource path, not the reader contract.
write_cues() {
  local output_yaml=$1
  local resource=$2
  cat > "${output_yaml}" <<EOF
cues:
  textual:
    type: json
    format: ${cue_format}
    guaranteed: true
    scope: speaker
    policy:
      type: fixed
      key: mix_spk_id
      resource: ${resource}
    collate:
      align: max
      pad_mode: constant
      fill_value: 0
      emit_present: false
EOF
}

# Keep all generated paths absolute so cue indexes can be reused by other recipes.
mkdir -p "${data}/${noise_type}"
output_root=$(realpath "${data}/${noise_type}")
for split in train-100 dev test; do
  output_dir="${output_root}/${split}"
  mkdir -p "${output_dir}/cues"
  python "${WESEP_ROOT}/examples/audio/librimix/local/scan_librimix.py" \
    "${mix_data_path}/${split}/mix_${noise_type}" \
    --outfile "${output_dir}/samples.jsonl"
  ln -sf samples.jsonl "${output_dir}/raw.list"

  # Raw records preserve the full transcript; word selection happens at runtime.
  python local/build_transcript_cues.py \
    --samples "${output_dir}/samples.jsonl" \
    --librispeech-root "${librispeech_root}" \
    --output "${output_dir}/cues/textual.json"

  # Offline and online tokenization use the same encoding configuration.
  if [ "${textual_input}" = "dae_phoneme" ]; then
    bash local/preprocess_textual.sh \
      --input "${output_dir}/cues/textual.json" \
      --output "${output_dir}/cues/textual_phoneme.json" \
      --config "${config}"
  fi

  # Full transcript indexes remain available independently of official cues.
  if [ "${split}" = test ]; then
    cue_yaml="${output_dir}/cues_full.yaml"
  else
    cue_yaml="${output_dir}/cues.yaml"
  fi
  write_cues "${cue_yaml}" "${output_dir}/cues/${cue_index}"
  if [ "${split}" = test ]; then
    for count in 1 2 3 4; do
      official="dae_kce/testset/kw-${count}_seed-42.jsonl"
      if [ ! -f "${official}" ]; then
        echo "Official cues not found: ${official}; run Stage 0 first." >&2
        exit 1
      fi
      index="${output_dir}/cues/textual_kw${count}.json"
      python local/dae_kce/convert_official_cues.py \
        --samples "${output_dir}/samples.jsonl" \
        --official-cues "${official}" \
        --textual-input "${textual_input}" --output "${index}"
      write_cues "${output_dir}/cues_kw${count}.yaml" "${index}"
    done
  fi
done

# Advertise the selected storage representation, as in the visual recipe.
python - "${output_root}/textual_cue.json" "${cue_format}" "${cue_index}" <<'PYTHON'
import json
import sys
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump({"type": "json", "format": sys.argv[2], "index": sys.argv[3]},
              stream, indent=2)
PYTHON
