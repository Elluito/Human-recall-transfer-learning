#!/usr/bin/env bash
# Run a transfer experiment: train a fresh agent on `to_task` while blending
# in Q-values from `from_task`'s pretrained checkpoint whenever the memory
# unit recognizes the current situation (paper Algorithm 1 / Equation 1).
#
# Usage: scripts/run_transfer.sh <from_task> <to_task> <num_training_episodes> [phi_version:1|2] [attempt_id]
#
# from_task's checkpoint must exist under pacman/models/ with the name in
# qlearningAgents.MODEL_REGISTRY, and its memory unit must exist at
# pacman/data/LTSM_AUTOENCODER_20000_task_<from_task>.h5 (the repo ships
# these for from_task in {0, 1}; see scripts/train_memory_unit.sh to build
# your own).
set -euo pipefail

FROM_TASK="${1:?usage: $0 <from_task> <to_task> <num_training_episodes> [phi_version] [attempt_id]}"
TO_TASK="${2:?usage: $0 <from_task> <to_task> <num_training_episodes> [phi_version] [attempt_id]}"
NUM_TRAINING="${3:?usage: $0 <from_task> <to_task> <num_training_episodes> [phi_version] [attempt_id]}"
PHI_VERSION="${4:-2}"
ATTEMPT="${5:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../pacman"

python pacman.py -p PacmanQAgent -g DirectionalGhost \
    -d "$TO_TASK" -x "$NUM_TRAINING" -n "$NUM_TRAINING" \
    --inicio "$ATTEMPT" --final "$ATTEMPT" \
    --transfer "${FROM_TASK},${TO_TASK}" \
    -s "LTSM_AUTOENCODER_20000_task_${FROM_TASK}.h5" \
    -a "phi_version=${PHI_VERSION}"

echo
echo "Win/score/epsilon/phi logs written to pacman/data/*_task_${TO_TASK}_*transfer_from_${FROM_TASK}*"
echo "Move them into results/<experiment-name>/ to plot with plot_results.py compare."
