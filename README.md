# Human-Like Recall/Association for Transfer Learning in Reinforcement Learning

Code for the Pacman transfer-learning experiments described in
[`pdf/Transfer_Learning_Pacman_section.pdf`](pdf/Transfer_Learning_Pacman_section.pdf)
(Edwin Duban Torres, Luis Alfredo Avendaño, Fernando Lozano; Universidad de
los Andes, February 2020).

**Idea:** an agent learns a sequence of tasks of increasing difficulty. A
"memory unit" -- an LSTM autoencoder trained on a previously learned task --
recognizes when the agent's current situation resembles that earlier task,
and when it does, the agent's action is chosen from a blend of the old
task's Q-values and the new task's Q-values instead of the new task's
Q-values alone. This directory contains everything needed to train the
tasks, train the memory unit, run the transfer experiments, and reproduce
the paper's figures.

## Repository layout

```
pdf/               the paper section this code implements
pacman/             runnable code (see pacman/README.md)
  pacman.py           CLI entry point / training driver (CS188 engine)
  qlearningAgents.py  DQN agent + transfer-learning logic (Algorithm 1)
  memory_unit.py       LSTM-autoencoder memory unit: train / evaluate
  plot_results.py      reproduces Figures 7-14
  data/                 working directory: checkpoints, logs, plots
  models/               Q-network checkpoints (not committed, see below)
  layouts/              auto-generated task layouts
results/            sample run data behind Figures 7-12 (see results/README.md)
scripts/            thin shell wrappers around the commands below
```

## Setup

Requires Python 3.8-3.11 (TensorFlow does not support newer interpreters as
of this writing).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU is optional: `tf.device("GPU:0")` calls in `qlearningAgents.py` fall
back to CPU automatically under TensorFlow 2's default soft device
placement, just slower (training a single task takes on the order of hours
on CPU).

## Quick start: reproduce the plots without training anything

`results/` ships the raw per-episode outcomes (10 runs per condition) behind
Figures 7-12, and `pacman/data/` ships the two trained memory units (task 0
and task 1). To regenerate the figures from that data:

```bash
scripts/plot_paper_figures.sh
```

This writes `results/figures/fig7_task1_phiV1.pdf` through
`fig12_task2_from_task0_phiV2.pdf`. See `results/README.md` for exactly
which directory backs which figure.

## Reproducing the experiments from scratch

All `pacman.py` commands below are also available as the wrapper scripts in
`scripts/`; both are shown so you can adapt flags directly if you deviate
from the paper's setup. Commands assume `pacman/` as the working directory
unless noted.

### 1. The environment

A 17x17 arena with one food pellet and one ghost (`DirectionalGhost`, which
chases Pacman). Reward is -1 per step, -500 if the ghost catches Pacman, and
+500 for reaching the food. Difficulty (`-d 0|1|2`) controls how many walls
surround the food (paper Figure 3); `pacman.py` regenerates the layout
procedurally before every episode via `crear_layout()`.

### 2. Train (or download) the baseline task checkpoints

Tasks 0 and 1 train for 25,000 episodes, task 2 for 40,000, with epsilon
decaying from 0.9 to 0.1 over the first half of training:

```bash
scripts/train_baseline.sh 0 25000   # task 0
scripts/train_baseline.sh 1 25000   # task 1
scripts/train_baseline.sh 2 40000   # task 2
```

equivalently:

```bash
cd pacman
python pacman.py -p PacmanQAgent -g DirectionalGhost -d 0 -x 25000 -n 25000
```

Checkpoints save to `pacman/models/` every 1000 episodes under an
auto-generated name (embedding a timestamp and hyperparameters). To use a
checkpoint later as a transfer source, rename/copy it to match the fixed
name `qlearningAgents.MODEL_REGISTRY` expects for that task (or edit
`MODEL_REGISTRY` to point at your generated filename). To skip training
tasks 0 and 1 yourself, download the two checkpoints used in the paper and
place them under `pacman/models/`; see `pacman/models/README.md` for the
links and expected filenames.

To reproduce the paper's 10 independent runs per condition, repeat a
command with a different `--inicio <attempt id>` each time (it's folded
into the output filenames), then move the resulting
`pacman/data/prob_task_*.txt` / `score_task_*.txt` files into
`results/<experiment-name>/` before plotting.

### 3. Train the memory unit

The memory unit needs its own training data: trajectories from a task's
*trained* agent, collected by playing games with training disabled
(`-x 0`, which also makes `runGames` load that task's `MODEL_REGISTRY`
checkpoint automatically):

```bash
scripts/collect_memory_data.sh 0 1000   # -> pacman/data/piezas_task_0
scripts/collect_memory_data.sh 1 1000   # -> pacman/data/piezas_task_1

scripts/train_memory_unit.sh 0   # -> pacman/data/LTSM_AUTOENCODER_20000_task_0.h5
scripts/train_memory_unit.sh 1   # -> pacman/data/LTSM_AUTOENCODER_20000_task_1.h5
```

equivalently:

```bash
cd pacman
python pacman.py -p PacmanQAgent -g DirectionalGhost -d 0 -x 0 -n 1000
python memory_unit.py train --task 0
python memory_unit.py evaluate --task 0
```

`train` fits the LSTM autoencoder (paper Figure 5: encoder LSTMs
64->32->16, a `RepeatVector` bottleneck, decoder LSTMs 16->32->64, and a
`TimeDistributed(Dense)` readout) on 18,000 of task 0's 25,000 collected
snapshots. `evaluate` reproduces the precision/recall-vs-threshold curve
(Figure 6) and the normalized confusion matrix (Tables 1-3) used to pick
`MEMORY_MSE_THRESHOLD = 0.02` (task 0 vs. task 1 as the "anomaly" class),
writing both plots to `pacman/data/`. The repository already ships trained
memory units for tasks 0 and 1, so this step is optional unless you want to
retrain them or inspect the precision/recall curves yourself.

### 4. Run a transfer experiment

```bash
scripts/run_transfer.sh 0 1 25000       # transfer task 0 -> task 1, phi V2 (default)
scripts/run_transfer.sh 1 2 40000       # transfer task 1 -> task 2
scripts/run_transfer.sh 0 2 40000 1     # transfer task 0 -> task 2, phi V1
```

equivalently:

```bash
cd pacman
python pacman.py -p PacmanQAgent -g DirectionalGhost \
    -d 1 -x 25000 -n 25000 \
    --transfer 0,1 -s LTSM_AUTOENCODER_20000_task_0.h5 \
    -a phi_version=2
```

`--transfer <from>,<to>` loads `<from>`'s checkpoint (from
`MODEL_REGISTRY`) as the source policy Q_old, alongside its memory unit
(`-s`, a filename under `pacman/data/`). At every step, if the memory unit's
reconstruction error for the last 4 timesteps of Pacman's 5x5 neighbourhood
is below `MEMORY_MSE_THRESHOLD`, the action comes from
`phi * Q_old + (1 - phi) * Q_new` instead of `Q_new` alone (Equation 1);
otherwise it's epsilon-greedy on `Q_new` as usual (Algorithm 1).

`-a phi_version=1|2` selects between the paper's two phi schedules (Section
4, Figure 4):

- **V1** (`phi_version=1`): phi starts at 0.8 and decays at a fixed
  per-episode rate (`PHI_DECAY = 0.999970043`), independent of the number of
  training episodes. Used for Figures 7-9.
- **V2** (`phi_version=2`, default): phi starts at 1 and the decay rate is
  solved so phi reaches 0.1 exactly at the last training episode. Used for
  Figures 10-12.

### 5. Plot results

```bash
cd ..   # repo root
python pacman/plot_results.py compare \
    --run "results/T1-exponencial-prob:T1 (no transfer)" \
    --run "results/T0-T1-low-phi:T0->T1" \
    --window 500 --ylabel "Winning probability" \
    --output results/figures/fig10_task1_phiV2.pdf
```

`compare` averages the 0/1 win indicator across a directory's `.txt` run
files, smooths with a moving average (`--window`, 500 in the paper),
shades +/-1 standard deviation across runs, and rescales the x-axis back to
the original episode count (smoothing shortens the series by `window - 1`
points). `scripts/plot_paper_figures.sh` runs this for all six of the
paper's comparison figures against the bundled sample data.

`plot_results.py heatmap` reproduces the agent-position heatmaps (Figures
13-14) from `pacman/data/history`, which `learningAgents.py` appends to
after every episode. Since a new training run truncates that file, rename
it after each run you want to keep, e.g.:

```bash
cd pacman
python pacman.py -p PacmanQAgent -g DirectionalGhost -d 0 -x 0 -n 1000  # uses task 0's checkpoint
mv data/history data/history_task0
cd ..
python pacman/plot_results.py heatmap \
    --run "pacman/data/history_task0:Task 0" \
    --output results/figures/fig14_heatmaps.pdf
```

## CLI reference (`pacman.py`)

| Flag | Meaning |
|---|---|
| `-p PacmanQAgent` | agent type (always this one for the transfer experiments) |
| `-g DirectionalGhost` | ghost policy; chases Pacman with probability `prob_attack` (default 0.8 -- the paper's "always moves to minimize its distance" simplifies this) |
| `-d 0\|1\|2` | task difficulty (paper Figure 3) |
| `-x N` | number of training episodes |
| `-n N` | total episodes to run (extra episodes beyond `-x` run the learned policy without further learning -- also how memory-unit data collection works, see step 3) |
| `--transfer FROM,TO` | enable transfer from task `FROM`'s checkpoint while training task `TO` |
| `-s FILENAME` | memory-unit checkpoint (under `pacman/data/`) required alongside `--transfer` |
| `-a phi_version=1\|2` | phi schedule (Section 4); other agent kwargs go here too, comma-separated |
| `--inicio N` | attempt id folded into output filenames, for running repeated trials |

## Implementation notes

This is research code kept close to its original form; a few things worth
knowing before you compare numbers against the paper:

- All output paths (`data/`, `models/`, `layouts/`) are relative to the
  current working directory -- always run from `pacman/` unless a command
  says otherwise.
- `DirectionalGhost`'s default `prob_attack=0.8` (80% of the time it takes
  the locally optimal chasing move, 20% a random legal move) is the closest
  built-in match to the paper's "the ghost always moves to minimize its
  distance to Pacman"; the CLI doesn't expose ghost kwargs, so exact
  determinism would require a small code change in `pacman.py`'s
  `runGames`.
- `qlearningAgents.MODEL_REGISTRY` hardcodes the exact filenames of the
  checkpoints referenced in the paper. Training your own task 0/1/2 agent
  produces a differently-named file (embedding a timestamp); rename it or
  edit the registry to reuse it as a transfer source.

## License and attribution

The Pacman game engine under `pacman/` (`pacman.py`, `game.py`,
`ghostAgents.py`, and friends) is adapted from the
[CS188 Pacman AI projects](http://ai.berkeley.edu) at UC Berkeley,
originally created by John DeNero and Dan Klein, with autograding by Brad
Miller, Nick Hay, and Pieter Abbeel; each such file keeps its original
license header (free for educational use, with attribution, without
redistributing course solutions). The transfer-learning agent, memory unit,
and plotting code are original contributions of this project, licensed
under the [MIT License](LICENSE).

If you use this code, please cite:

> E. Duban Torres, L. A. Avendaño, F. Lozano. "Human-Like
> Recall/Association for Transfer Learning in Reinforcement Learning."
> Universidad de los Andes, February 2020.
