# pacman/

This directory is a runnable Python package: a modified copy of the
[UC Berkeley CS188](http://ai.berkeley.edu) Pacman AI projects framework,
extended with the transfer-learning agent, memory unit, and plotting tools
described in `../pdf/Transfer_Learning_Pacman_section.pdf`. See the
top-level README for setup and reproduction instructions -- everything here
is meant to be run with this directory (`pacman/`) as the working directory,
e.g. `cd pacman && python pacman.py ...`.

## Files

**Original CS188 engine** (game loop, layouts, display, generic RL
scaffolding -- kept close to upstream, see each file's header for the
original license/attribution):
`pacman.py`, `game.py`, `ghostAgents.py`, `keyboardAgents.py`,
`pacmanAgents.py`, `learningAgents.py`, `mdp.py`, `featureExtractors.py`,
`layout.py`, `util.py`, `graphicsDisplay.py`, `graphicsUtils.py`,
`textDisplay.py`.

**This project's contribution** (paper Sections 3-6):
- `qlearningAgents.py` -- the DQN agent (`Policy`, `QLearningAgent`,
  `PacmanQAgent`) and the transfer-learning logic: when the memory unit
  flags the current situation as similar to a source task, the action is
  chosen from a phi-weighted blend of the source and target Q-values
  (Algorithm 1 / Equation 1). `segtree.py` backs its (unused by default)
  prioritized replay buffer.
- `memory_unit.py` -- trains and evaluates the LSTM-autoencoder memory unit
  (Section 5, Figures 5-6, Tables 1-3).
- `plot_results.py` -- reproduces the learning-curve comparison plots
  (Figures 7-12) and agent-position heatmaps (Figures 13-14) from Section 6.

**Working directories** (see their own READMEs): `data/`, `models/`,
`layouts/`.

## Licensing note

The CS188 engine files retain their original per-file license header:
free to use/extend for educational purposes, with attribution to UC
Berkeley and a link to http://ai.berkeley.edu, and without redistributing
CS188 course solutions. `qlearningAgents.py`, `memory_unit.py`, and
`plot_results.py` are original contributions of this project (see the
top-level `LICENSE`), built on top of that same CS188 scaffolding.
