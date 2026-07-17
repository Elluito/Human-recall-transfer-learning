#!/usr/bin/env bash
# Train a task from scratch, with no transfer (the "T<N>" baseline curves in
# the paper, e.g. Figures 7-12's blue lines).
#
# Usage: scripts/train_baseline.sh <difficulty:0|1|2> <num_training_episodes> [num_games]
#
# num_games defaults to num_training_episodes (pure training, no extra
# "prueba"/evaluation episodes at the end). Pass a larger num_games if you
# also want to collect memory-unit training data for this task right after
# training -- see scripts/collect_memory_data.sh for the recommended way to
# do that against a saved checkpoint instead.
set -euo pipefail

DIFFICULTY="${1:?usage: $0 <difficulty:0|1|2> <num_training_episodes> [num_games]}"
NUM_TRAINING="${2:?usage: $0 <difficulty:0|1|2> <num_training_episodes> [num_games]}"
NUM_GAMES="${3:-$NUM_TRAINING}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../pacman"

python pacman.py -p PacmanQAgent -g DirectionalGhost \
    -d "$DIFFICULTY" -x "$NUM_TRAINING" -n "$NUM_GAMES"

echo
echo "Checkpoints are saved under pacman/models/ every 1000 episodes."
echo "To use this checkpoint as a transfer source or for --numTraining 0"
echo "evaluation, rename/copy the newest one to match qlearningAgents.py's"
echo "MODEL_REGISTRY[$DIFFICULTY] (or edit MODEL_REGISTRY to point at it)."
