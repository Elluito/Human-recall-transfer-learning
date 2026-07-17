# Results

This directory contains the raw per-episode outcomes from the training runs
behind Section 6 of the paper (`../pdf/Transfer_Learning_Pacman_section.pdf`).
Each subdirectory holds one `.txt` file per independent run (the paper uses
10 runs per condition); "prob" files contain a `0`/`1` win indicator per
episode, "score" files contain the raw Pacman score per episode.

Regenerate any of the paper's figures from these files with
`../pacman/plot_results.py compare` (see the top-level README for the exact
commands used for Figures 7-12). To reproduce a directory from scratch
instead of using the copies here, see "Reproducing the experiments" in the
top-level README.

## Directory -> experiment mapping

| Directory | Task | Condition | Phi version |
|---|---|---|---|
| `T1-exponencial-prob` / `T1-exponencial-score` | Task 1 | No transfer (baseline) | n/a |
| `T2-exponencial-prob` / `T2-exponencial-score` | Task 2 | No transfer (baseline) | n/a |
| `T0-T1-exponencial-prob` | Task 1 | Transfer from task 0 | V1 |
| `T0-T1-low-phi` | Task 1 | Transfer from task 0 | V2 |
| `T1-T2-exponencial-prob` / `T1-T2-exponencial-score` | Task 2 | Transfer from task 1 | V1 |
| `T1-T2-low-phi` | Task 2 | Transfer from task 1 | V2 |
| `T0-T2-prob` / `T0-T2-score` | Task 2 | Transfer from task 0 | V1 |
| `T0-T2-low-phi` | Task 2 | Transfer from task 0 | V2 |
| `T0-T2-esquina_sup_der-prob` | Task 2 | Transfer from task 0 (alternate food-corner run) | V1 |
| `Epsilon`, `Phi` | -- | Recorded epsilon/phi decay traces used for Figure 4 | -- |

"V1"/"V2" are the two phi schedules from paper Section 4 (see
`phi_version` in `../pacman/qlearningAgents.py`). Paper figure numbers:

| Figure | Baseline (no transfer) | Transfer |
|---|---|---|
| 7 (task 1, Phi V1) | `T1-exponencial-prob` | `T0-T1-exponencial-prob` |
| 8 (task 2, Phi V1) | `T2-exponencial-prob` | `T1-T2-exponencial-prob` |
| 9 (task 2, Phi V1) | `T2-exponencial-prob` | `T0-T2-prob` |
| 10 (task 1, Phi V2) | `T1-exponencial-prob` | `T0-T1-low-phi` |
| 11 (task 2, Phi V2) | `T2-exponencial-prob` | `T1-T2-low-phi` |
| 12 (task 2, Phi V2) | `T2-exponencial-prob` | `T0-T2-low-phi` |

See the top-level README for the exact `plot_results.py compare` commands
that reproduce each row.

## `figures/`

Reference PDFs regenerated from the tables above via
`../scripts/plot_paper_figures.sh` -- a quick way to see the expected output
without running anything. They're build artifacts, not source data; rerun
that script any time to refresh them.
