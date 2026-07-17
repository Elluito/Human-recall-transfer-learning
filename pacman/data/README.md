# data/

Working directory for generated and downloaded artifacts. `pacman.py`,
`qlearningAgents.py`, `memory_unit.py`, and `learningAgents.py` all read and
write here using paths relative to `pacman/` (so run all commands from
inside this directory, as the top-level README does).

Shipped in the repository:

- `LTSM_AUTOENCODER_20000_task_0.h5`, `LTSM_AUTOENCODER_20000_task_1.h5` --
  pretrained memory units (LSTM autoencoders, paper Section 5) for tasks 0
  and 1, used by the `--transfer` experiments out of the box.

Generated at runtime (not committed):

- `piezas_task_<N>` -- pickled memory-unit training examples collected by a
  `PacmanQAgent` in data-collection mode.
- `history` -- pickled per-episode Pacman position trails, consumed by
  `plot_results.py heatmap`.
- `prob_task_*.txt`, `score_task_*.txt`, `epsilon_para_*.txt`,
  `phi_para_*.txt` -- per-run training logs written by `pacman.py`'s
  `runGames`. Copy/move these into `../results/<experiment>/` to keep them
  across runs before the next run overwrites `history`/etc.
- `*.pdf`, `*.png` -- plots written by `memory_unit.py`.
