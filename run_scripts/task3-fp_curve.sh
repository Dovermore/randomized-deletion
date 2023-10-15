#!/usr/bin/env bash
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    echo "This script should only be sourced not executed directly"
    exit 1
fi

ckpts=(
    outputs/models/checkpoint/malconv-byte_deletion_99.5_sd_42.ckpt
    outputs/models/checkpoint/malconv-insn_deletion_99.5_sd_42.ckpt
    outputs/models/checkpoint/malconv-original_sd_42.ckpt
)

configs=(
    configs/repeat-forward-exp/malconv-byte_deletion_99.5-header-50.yaml
    configs/repeat-forward-exp/malconv-insn_deletion_99.5-header-50.yaml
    configs/repeat-forward-exp/malconv-original-header-1.yaml
)

task=fp_curve-repeat_forward
jobs=()
for i in "${!ckpts[@]}"
do
    job="python3 /app/src/$task.py --path ${ckpts[@]:${i}:1} --repeat-conf ${configs[@]:${i}:1} --num-partitions 1 --num-pred 10"
    jobs+=("$job")
done
