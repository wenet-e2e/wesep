#!/usr/bin/env bash

set -euo pipefail

config=confs/simulation/respeaker_4mic_16k.yaml
librimix_root=
output_root=
split=all

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

if [ -z "${librimix_root}" ] || [ -z "${output_root}" ]; then
  echo "Both --librimix-root and --output-root are required." >&2
  exit 1
fi

# Rebuild the deterministic scenarios and detect configuration changes.
scenario_manifest="${output_root}/metadata/scenarios_v1.csv.gz"
scenario_candidate="${scenario_manifest%.csv.gz}.new.csv.gz"
python local/simulation/create_scenarios.py \
  --config "${config}" \
  --librimix-root "${librimix_root}" \
  --output "${scenario_candidate}"

overwrite=--overwrite
if [ -f "${scenario_manifest}" ] && cmp -s \
    "${scenario_candidate}" "${scenario_manifest}"; then
  overwrite=
fi
mv "${scenario_candidate}" "${scenario_manifest}"

# Generate all requested waveforms from the fixed scenarios.
python local/simulation/simulate_librimix.py \
  --config "${config}" \
  --librimix-root "${librimix_root}" \
  --scenario-manifest "${scenario_manifest}" \
  --output-root "${output_root}" \
  --split "${split}" \
  ${overwrite}

# Export only the spatial fields required by the WeSep cue interface.
python local/build_spatial_cues.py \
  --scenario-manifest "${scenario_manifest}" \
  --output-root "${output_root}" \
  --split "${split}"
