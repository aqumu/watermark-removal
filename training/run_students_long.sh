#!/usr/bin/env bash
# Continue students from their first-run weights (training/runs/teachers/student_<s>_best.pth).
set -u
cd "$(dirname "$0")/.."
for s in "$@"; do
    echo "=== student $s long $(date +%T)"
    .venv/Scripts/python -u training/train_overlay.py --config "training/configs/cian_student_${s}_long.yaml" \
        --load-weights "training/runs/teachers/student_${s}_best.pth" \
        > "training/runs/student_${s}_long.out" 2>&1 || echo "student $s failed"
done
echo "=== done $(date +%T)"
