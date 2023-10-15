#!/usr/bin/env bash
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    echo "This script should only be sourced not executed directly"
    exit 1
fi

configs=(
    configs/models/malconv-byte_deletion_99.5-header.yaml
    configs/models/malconv-insn_deletion_99.5-header.yaml
    configs/models/malconv-original-header.yaml
)

jobs=()
task=train
for idx in "${configs[@]}"; do
    job="python3 /app/src/$task.py --config $idx"
    jobs+=("$job")
done
