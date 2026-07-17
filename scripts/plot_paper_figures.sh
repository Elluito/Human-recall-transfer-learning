#!/usr/bin/env bash
# Regenerate paper Figures 7-12 from the sample run data shipped in
# results/ (see results/README.md for the directory -> figure mapping).
# No training required -- this only reads the bundled .txt logs.
#
# Usage: scripts/plot_paper_figures.sh [output_dir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$REPO_ROOT/results/figures}"
mkdir -p "$OUT_DIR"

cd "$REPO_ROOT"

python pacman/plot_results.py compare --window 500 --ylabel "Winning probability" \
    --title "Task 1, Phi V1" \
    --run "results/T1-exponencial-prob:T1 (no transfer)" \
    --run "results/T0-T1-exponencial-prob:T0->T1" \
    --output "$OUT_DIR/fig7_task1_phiV1.pdf"

python pacman/plot_results.py compare --window 500 --ylabel "Winning probability" \
    --title "Task 2, Phi V1" \
    --run "results/T2-exponencial-prob:T2 (no transfer)" \
    --run "results/T1-T2-exponencial-prob:T1->T2" \
    --output "$OUT_DIR/fig8_task2_from_task1_phiV1.pdf"

python pacman/plot_results.py compare --window 500 --ylabel "Winning probability" \
    --title "Task 2, Phi V1" \
    --run "results/T2-exponencial-prob:T2 (no transfer)" \
    --run "results/T0-T2-prob:T0->T2" \
    --output "$OUT_DIR/fig9_task2_from_task0_phiV1.pdf"

python pacman/plot_results.py compare --window 500 --ylabel "Winning probability" \
    --title "Task 1, Phi V2" \
    --run "results/T1-exponencial-prob:T1 (no transfer)" \
    --run "results/T0-T1-low-phi:T0->T1" \
    --output "$OUT_DIR/fig10_task1_phiV2.pdf"

python pacman/plot_results.py compare --window 500 --ylabel "Winning probability" \
    --title "Task 2, Phi V2" \
    --run "results/T2-exponencial-prob:T2 (no transfer)" \
    --run "results/T1-T2-low-phi:T1->T2" \
    --output "$OUT_DIR/fig11_task2_from_task1_phiV2.pdf"

python pacman/plot_results.py compare --window 500 --ylabel "Winning probability" \
    --title "Task 2, Phi V2" \
    --run "results/T2-exponencial-prob:T2 (no transfer)" \
    --run "results/T0-T2-low-phi:T0->T2" \
    --output "$OUT_DIR/fig12_task2_from_task0_phiV2.pdf"

echo
echo "Figures written to $OUT_DIR"
