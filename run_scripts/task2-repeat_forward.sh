#!/usr/bin/env bash
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    echo "This script should only be sourced not executed directly"
    exit 1
fi

configs=(
    configs/repeat-forward-exp/malconv-byte_deletion_99.5-header-50.yaml
    configs/repeat-forward-exp/malconv-insn_deletion_99.5-header-50.yaml
    configs/repeat-forward-exp/malconv-original-header-1.yaml
)

task=repeat_forward_exp
jobs=()
for idx in "${!configs[@]}"; do
    job="python3 /app/src/$task.py --config ${configs[@]:${idx}:1} --partition 0 --num-partitions 1"
    jobs+=("$job")
done
