#!/usr/bin/env bash

set -euo pipefail

if [ -z "${WESEP_ROOT:-}" ]; then
  . "$(dirname "${BASH_SOURCE[0]}")/../../path.sh"
fi

. "${WESEP_ROOT}/tools/parse_options.sh" || exit 1

# Official DAE resources always live under this recipe.
output_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/dae_kce"
mkdir -p "${output_dir}"

# Download one immutable input only when it is absent locally.
download_if_missing() {
  local url=$1
  local output=$2
  if [ ! -f "${output}" ]; then
    wget -O "${output}" "${url}"
  fi
}

# KCE weights and architecture files are published by the DAE-TSE authors.
hf_root=https://huggingface.co/GnafiY/DAE-TSE/resolve/main/kce
download_if_missing "${hf_root}/epoch_149.pt" "${output_dir}/epoch_149.pt"
download_if_missing "${hf_root}/model.yaml" "${output_dir}/model.yaml"
download_if_missing "${hf_root}/data.yaml" "${output_dir}/data.yaml"

# Text resources match the KCE checkpoint phoneme vocabulary.
github_root=https://raw.githubusercontent.com/GnafiY/DAE-TSE/main/recipes/librimix/dae-tse/data/text_cue
download_if_missing "${github_root}/phoneme2int.txt" \
  "${output_dir}/phoneme2int.txt"
download_if_missing "${github_root}/word2lexicon.txt" \
  "${output_dir}/word2lexicon.txt"

# Keep the four official evaluation conditions unchanged.
mkdir -p "${output_dir}/testset"
for count in 1 2 3 4; do
  download_if_missing "${github_root}/testset/kw-${count}_seed-42.jsonl" \
    "${output_dir}/testset/kw-${count}_seed-42.jsonl"
done
