"""
plot_results.py
----------------
Reproduces the result plots from Section 6 of the paper
(pdf/Transfer_Learning_Pacman_section.pdf): win-probability learning curves
with and without transfer (Figures 7-12), and agent-position heatmaps
(Figures 13-14).

Each pacman.py training run writes one .txt file per attempt (one float per
line: 0/1 win indicator for "prob" files, raw score for "score" files) into
a directory such as results/T0-T1-low-phi/. The paper's protocol runs each
condition 10 times and reports the across-run mean with a shaded +/- 1
standard deviation band, both smoothed with a moving average (default
window 500); this script reproduces that exactly, including rescaling the
x-axis back to the original episode count after smoothing shortens the
series (see Section 6's note about this).

Examples
--------
Figure 7 (task 1, with vs. without transfer from task 0):
    python plot_results.py compare \\
        --run "../results/T1-exponencial-prob:T1 (no transfer)" \\
        --run "../results/T0-T1-low-phi:T0->T1" \\
        --window 500 --ylabel "Winning probability" \\
        --output ../results/fig7_task1_transfer.pdf

Figure 14 (position heatmaps for tasks 0, 1, and 2 with transfer):
    python plot_results.py heatmap \\
        --run "data/history_task0:Task 0" \\
        --run "data/history_task1:Task 1" \\
        --run "data/history_task2_transfer:Task 2 (T1->T2)" \\
        --output ../results/fig14_heatmaps.pdf
"""
import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def _split_run_spec(spec):
    """Parse a "path:label" CLI argument; label defaults to the directory name."""
    path, _, label = spec.partition(":")
    if not label:
        label = os.path.basename(path.rstrip("/")) or path
    return path, label


def load_run_directory(directory):
    """Load every *.txt run file in `directory` as columns of one 2D array
    (episodes x independent runs), matching the paper's "10 independent
    runs, averaged" protocol."""
    files = sorted(glob.glob(os.path.join(directory, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt run files found in {directory}")
    columns = [np.loadtxt(f) for f in files]
    min_len = min(len(c) for c in columns)
    return np.column_stack([c[:min_len] for c in columns])


def running_mean(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def smooth_runs(data, window):
    """Apply a moving average of size `window` to each run (column), then
    return the per-episode mean and standard deviation across runs."""
    smoothed = np.column_stack([running_mean(data[:, i], window) for i in range(data.shape[1])])
    return smoothed.mean(axis=1), smoothed.std(axis=1)


def plot_mean_and_ci(ax, x, mean, std, label, color):
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.3)
    ax.plot(x, mean, color=color, label=label)


def compare(args):
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, spec in enumerate(args.run):
        directory, label = _split_run_spec(spec)
        data = load_run_directory(directory)
        n_episodes = data.shape[0]
        mean, std = smooth_runs(data, args.window)
        # Smoothing shortens the series by (window - 1) points; rescale the
        # x-axis so it still reads in terms of the original episode count.
        x = np.linspace(0, n_episodes, len(mean))
        plot_mean_and_ci(ax, x, mean, std, label, f"C{idx}")

    ax.set_xlabel("Episodes", fontsize=16)
    ax.set_ylabel(args.ylabel, fontsize=16)
    ax.set_title(args.title or f"Estimate with a moving average of size {args.window}", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(prop={"size": 14})
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Saved {args.output}")


def load_history(path):
    """Load a data/history-style pickle: one entry per episode, each a list
    of Pacman (x, y) position arrays visited during that episode (written by
    ReinforcementAgent.final in learningAgents.py)."""
    episodes = []
    with open(path, "rb") as fp:
        while True:
            try:
                episodes.append(pickle.load(fp))
            except EOFError:
                break
    return episodes


def position_grid(episodes, grid_size):
    grid = np.zeros((grid_size, grid_size))
    for positions in episodes:
        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < grid_size and 0 <= y < grid_size:
                grid[x, y] += 1
    return grid


def heatmap(args):
    n = len(args.run)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    axes = axes[0]
    for ax, spec in zip(axes, args.run):
        path, label = _split_run_spec(spec)
        episodes = load_history(path)
        grid = position_grid(episodes, args.grid_size)
        im = ax.imshow(np.rot90(grid))
        ax.set_title(label, fontsize=16)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Saved {args.output}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_compare = sub.add_parser("compare", help="Win-probability / score comparison plot (Figures 7-12).")
    p_compare.add_argument("--run", action="append", required=True,
                            help='"directory:label", repeatable. directory holds one .txt file per run.')
    p_compare.add_argument("--window", type=int, default=500, help="Moving-average window (paper uses 500).")
    p_compare.add_argument("--ylabel", default="Winning probability")
    p_compare.add_argument("--title", default=None)
    p_compare.add_argument("--output", required=True)
    p_compare.set_defaults(func=compare)

    p_heatmap = sub.add_parser("heatmap", help="Agent position heatmap(s) (Figures 13-14).")
    p_heatmap.add_argument("--run", action="append", required=True,
                            help='"history_file:label", repeatable. One subplot per entry.')
    p_heatmap.add_argument("--grid-size", type=int, default=19,
                            help="Side length of the (square) Pacman arena grid, walls included.")
    p_heatmap.add_argument("--output", required=True)
    p_heatmap.set_defaults(func=heatmap)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.func(args)
