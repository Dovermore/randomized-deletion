#!/usr/bin/env bash
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    echo "This script should only be sourced not executed directly"
    exit 1
fi
jobs=()

configs=(
    configs/models/malconv-byte_deletion_99.5-header.yaml
    configs/models/malconv-insn_deletion_99.5-header.yaml
    configs/models/malconv-original-header.yaml
)

task=train
for idx in "${configs[@]}"; do
    job="python3 /app/src/$task.py --config $idx"
    jobs+=("$job")
done

configs=(
    configs/repeat-forward-exp/malconv-byte_deletion_99.5-header-50.yaml
    configs/repeat-forward-exp/malconv-insn_deletion_99.5-header-50.yaml
    configs/repeat-forward-exp/malconv-original-header-1.yaml
)

task=repeat_forward_exp
for idx in "${!configs[@]}"; do
    job="python3 /app/src/$task.py --config ${configs[@]:${idx}:1} --partition 0 --num-partitions 1"
    jobs+=("$job")
done


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
for i in "${!ckpts[@]}"
do
    job="python3 /app/src/$task.py --path ${ckpts[@]:${i}:1} --repeat-conf ${configs[@]:${i}:1} --num-partitions 1 --num-pred 10"
    jobs+=("$job")
done

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
for idx in "${!certify_configs[@]}"; do
    job="python3 /app/src/$task.py --repeat-conf ${repeat_configs[@]:${idx}:1} --certify-conf ${certify_configs[@]:${idx}:1} --num-partitions 1 --fp-ratio 0.005"
    jobs+=("$job")
done
