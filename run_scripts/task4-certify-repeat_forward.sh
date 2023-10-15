#!/usr/bin/env bash
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    echo "This script should only be sourced not executed directly"
    exit 1
fi

repeat_configs=(
    configs/repeat-forward-exp/malconv-byte_deletion_99.5-header-50.yaml
    configs/repeat-forward-exp/malconv-insn_deletion_99.5-header-50.yaml
    configs/repeat-forward-exp/malconv-original-header-1.yaml
)
certify_configs=(
    configs/certify-exp/malconv-byte_deletion_99.5-header-0.05-10-40.yaml
    configs/certify-exp/malconv-insn_deletion_99.5-header-0.05-10-40.yaml
    configs/certify-exp/malconv-original-header-0.05-1-0.yaml
)

task=certify_exp-repeat_forward
jobs=()
for idx in "${!certify_configs[@]}"; do
    job="python3 /app/src/$task.py --repeat-conf ${repeat_configs[@]:${idx}:1} --certify-conf ${certify_configs[@]:${idx}:1} --num-partitions 1 --fp-ratio 0.005"
    jobs+=("$job")
done
