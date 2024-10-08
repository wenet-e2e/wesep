#!/usr/bin/env bash
mindepth=0
maxdepth=1

. tools/parse_options.sh

if [ $# -gt 1 ]; then
    echo "Usage: $0 --mindepth 0 --maxdepth 1 [exp]" 1>&2
    echo ""
    echo "Show the system environments and the evaluation results in Markdown format."
    echo 'The default of <exp> is "exp/".'
    exit 1
fi

[ -f ./path.sh ] && . ./path.sh
set -euo pipefail
if [ $# -eq 1 ]; then
    exp=$(realpath "$1")
else
    exp=exp
fi

cat <<EOF
<!-- Generated by $0 -->
# RESULTS
## Environments
- date: \`$(LC_ALL=C date)\`
EOF

cat <<EOF
- Git hash: \`$(git rev-parse HEAD)\`
  - Commit date: \`$(git log -1 --format='%cd')\`

EOF

while IFS= read -r expdir; do
    if ls "${expdir}"/*/scoring/result_stoi.txt &>/dev/null; then
        echo -e "\n## $(basename ${expdir})\n"
        [ -e "${expdir}"/config.yaml ] && grep ^config "${expdir}"/config.yaml
        metrics=()
        heading="\n|dataset|"
        sep="|---|"
        for type in pesq pesq_wb pesq_nb estoi stoi sar sdr sir si_snr ovrl sig bak p808_mos; do
            if ls "${expdir}"/*/scoring/result_${type}.txt &>/dev/null; then
                metrics+=("$type")
                heading+="${type^^}|"
                sep+="---|"
            fi
        done
        echo -e "${heading}\n${sep}"

        setnames=()
        for dirname in "${expdir}"/*/scoring/result_stoi.txt; do
            dset=$(echo $dirname | sed -e "s#${expdir}/\([^/]*\)/scoring/result_stoi.txt#\1#g")
            setnames+=("$dset")
        done
        for dset in "${setnames[@]}"; do
            line="|${dset}|"
            for ((i = 0; i < ${#metrics[@]}; i++)); do
                type=${metrics[$i]}
                if [ -f "${expdir}"/${dset}/scoring/result_${type}.txt ]; then
                    score=$(head -n1 "${expdir}"/${dset}/scoring/result_${type}.txt)
                else
                    score=""
                fi
                line+="${score}|"
            done
            echo $line
        done
        echo ""
    fi

done < <(find ${exp} -mindepth ${mindepth} -maxdepth ${maxdepth} -type d)
