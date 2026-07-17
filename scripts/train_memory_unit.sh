#!/usr/bin/env bash
# Train and evaluate the memory unit (LSTM autoencoder, paper Section 5) for
# one task, using the data collected by scripts/collect_memory_data.sh.
# Reproduces paper Figure 5's model, Figure 6's precision/recall curve, and
# the confusion matrix in Tables 1-3.
#
# Usage: scripts/train_memory_unit.sh <task:0|1> [epochs]
set -euo pipefail

TASK="${1:?usage: $0 <task:0|1> [epochs]}"
EPOCHS="${2:-200}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../pacman"

python memory_unit.py train --task "$TASK" --epochs "$EPOCHS"
python memory_unit.py evaluate --task "$TASK"

echo
echo "Saved pacman/data/LTSM_AUTOENCODER_20000_task_${TASK}.h5"
echo "Precision/recall curve and confusion matrix plots are in pacman/data/"
