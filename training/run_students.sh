#!/usr/bin/env bash
# Train the distilled students one after another (each needs the teacher checkpoint at
# training/runs/teachers/teacher_ft3_best.pth). Usage: bash training/run_students.sh a b c d
set -u
cd "$(dirname "$0")/.."
for s in "$@"; do
    echo "=== student $s $(date +%T)"
    .venv/Scripts/python -u training/train_overlay.py --config "training/configs/cian_student_$s.yaml" \
        > "training/runs/student_$s.out" 2>&1 || echo "student $s failed"
done
echo "=== done $(date +%T)"
