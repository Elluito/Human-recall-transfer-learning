"""
memory_unit.py
--------------
Trains and evaluates the "memory unit" described in Section 5 of the paper
(pdf/Transfer_Learning_Pacman_section.pdf): an LSTM autoencoder that
recognizes whether the agent's current situation -- a short temporal
sequence of 5x5 neighbourhoods centred on Pacman -- resembles a previously
learned task. qlearningAgents.py loads the resulting model as
`similarity_function` and uses its reconstruction error at inference time to
decide when to blend in the source task's Q-values (see
MEMORY_MSE_THRESHOLD there, and Algorithm 1 / Equation 1 in the paper).

Data pipeline
-------------
1. Run pacman.py with a trained PacmanQAgent in data-collection mode (see
   scripts/collect_memory_data.sh) to populate data/piezas_task_<N>: each
   entry is a flattened (memory_length x 27) sequence of 5x5-neighbourhood
   snapshots plus Pacman's (x, y) position.
2. `python memory_unit.py train --task N` trains an autoencoder on task N's
   data and saves it to data/LTSM_AUTOENCODER_20000_task_<N>.h5. The
   repository ships pretrained copies for N=0 and N=1 (the two memories used
   in the paper's T0->T1, T1->T2 and T0->T2 transfer experiments).
3. `python memory_unit.py evaluate --task N` reproduces the precision/recall
   curve (paper Figure 6) and the normalized confusion matrix (paper Tables
   1-3) that were used to pick MEMORY_MSE_THRESHOLD = 0.02, treating task
   N+1 as the "anomaly" (not-similar) class.
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

seed(7)

N_TASKS = 3
MEMORY_LENGTH = 4   # timesteps; must match QLearningAgent.memory_length
N_FEATURES = 27     # 5x5 neighbourhood (25 cells) + Pacman's (x, y)
MEMORY_MSE_THRESHOLD = 0.02


def load_task_data(data_dir="data", tasks=None):
    """Load the pickled memory-unit examples collected per task (see
    QLearningAgent.getAction's data-collection branch in qlearningAgents.py).
    `tasks` defaults to all of them; pass an explicit list to avoid
    requiring files for tasks you don't need."""
    dataset = {}
    for task in (tasks if tasks is not None else range(N_TASKS)):
        objects = []
        filename = f"{data_dir}/piezas_task_{task}"
        with open(filename, "rb") as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
        dataset[task] = objects
    return dataset


def flatten(X):
    """Collapse a (samples, timesteps, features) 3D array to the last
    timestep of each sample, giving a (samples, features) 2D array."""
    flattened_X = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, X.shape[1] - 1, :]
    return flattened_X


def scale(X, scaler):
    """Apply a fitted sklearn scaler to every timestep of a 3D array."""
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X


def _prepare_examples(raw_examples, time_step=MEMORY_LENGTH, n_features=N_FEATURES):
    """Reshape the flat pickled examples to (time_step, n_features) and
    standardize each one independently (zero mean, unit variance per
    feature column), matching the preprocessing used when the shipped
    LTSM_AUTOENCODER_*.h5 checkpoints were trained."""
    prepared = [
        preprocessing.scale(np.asarray(x).reshape(time_step, n_features))
        for x in raw_examples
    ]
    return np.array(prepared)


def _split_own_task(X_own):
    """Reproduce the original train/val/test split: 10% held out as test,
    then 20% of the remainder as validation. For n_train=25000 examples
    this leaves an 18000-example training split, matching the "18000
    examples labeled 0" test set reported in the paper's Section 5."""
    X_train, X_test = train_test_split(X_own, test_size=0.1)
    X_train, X_val = train_test_split(X_train, test_size=0.2)
    return X_train, X_val, X_test


def build_autoencoder(time_step=MEMORY_LENGTH, n_features=N_FEATURES):
    """LSTM autoencoder matching Figure 5 of the paper: encoder LSTMs
    (4,64)->(4,32)->(1,16), a RepeatVector bottleneck, and decoder LSTMs
    (4,16)->(4,32)->(4,64) followed by a TimeDistributed(Dense) readout
    back to (4,27)."""
    model = keras.Sequential([
        LSTM(64, activation="relu", input_shape=(time_step, n_features), return_sequences=True),
        LSTM(32, activation="relu", return_sequences=True),
        LSTM(16, activation="relu", return_sequences=False),
        RepeatVector(time_step),
        LSTM(16, activation="relu", return_sequences=True),
        LSTM(32, activation="relu", return_sequences=True),
        LSTM(64, activation="relu", return_sequences=True),
        TimeDistributed(Dense(n_features)),
    ])
    model.compile(loss="mse", optimizer=optimizers.Adam(0.0001))
    return model


def train_memory_unit(task, data_dir="data", n_train=25000, batch_size=128, epochs=200):
    """Train the memory unit for `task` and save it to
    <data_dir>/LTSM_AUTOENCODER_20000_task_<task>.h5 (paper Section 5)."""
    data = load_task_data(data_dir, tasks=[task])
    X_own = _prepare_examples(data[task][:n_train])
    X_own = shuffle(X_own)
    X_train, X_val, _X_test = _split_own_task(X_own)

    model = build_autoencoder()
    history = model.fit(
        X_train, X_train,
        batch_size=batch_size, epochs=epochs, verbose=2,
        validation_data=(X_val, X_val),
    )

    plt.figure()
    plt.plot(history.history["loss"], linewidth=2, label="Train")
    plt.plot(history.history["val_loss"], linewidth=2, label="Valid")
    plt.legend(loc="upper right")
    plt.title("Memory unit training loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{data_dir}/memory_unit_task_{task}_loss.png")
    plt.close()

    out_path = f"{data_dir}/LTSM_AUTOENCODER_20000_task_{task}.h5"
    model.save(out_path)
    print(f"Saved memory unit for task {task} to {out_path}")
    return model


def evaluate_memory_unit(task, data_dir="data", threshold=MEMORY_MSE_THRESHOLD,
                          n_train=25000, n_eval=1000):
    """Reproduce paper Figure 6 (precision/recall vs threshold) and Tables
    1-3 (normalized confusion matrix) for the memory unit trained on
    `task`, treating `task + 1` as the anomaly (not-similar) class."""
    if task + 1 >= N_TASKS:
        raise ValueError(f"No task {task + 1} to use as the anomaly class for task {task}.")

    data = load_task_data(data_dir, tasks=[task, task + 1])
    X_own = _prepare_examples(data[task][:n_train])
    X_own = shuffle(X_own)
    X_own_train, _X_own_val, _X_own_test = _split_own_task(X_own)

    X_next = _prepare_examples(data[task + 1][:n_eval])
    X_next = shuffle(X_next)

    model = keras.models.load_model(f"{data_dir}/LTSM_AUTOENCODER_20000_task_{task}.h5")

    X = np.vstack((X_own_train, X_next))
    y = np.array([0] * X_own_train.shape[0] + [1] * X_next.shape[0])

    reconstructed = model.predict(X)
    mse = np.mean(np.power(flatten(X) - flatten(reconstructed), 2), axis=1)

    plt.figure(figsize=(10, 10))
    precision, recall, thresholds = precision_recall_curve(y, mse)
    plt.plot(thresholds, precision[1:], label="Precision", linewidth=2)
    plt.plot(thresholds, recall[1:], label="Recall", linewidth=2)
    plt.title("Precision and recall vs threshold", fontsize=15)
    plt.xlabel("Threshold", fontsize=20)
    plt.ylabel("Precision/Recall", fontsize=20)
    plt.legend(prop={"size": 20})
    plt.savefig(f"{data_dir}/precision_recall_task_{task}.pdf")
    plt.close()

    y_pred = mse > threshold
    matrix = confusion_matrix(y, y_pred, normalize="true")

    plt.matshow(matrix, cmap="autumn")
    plt.xticks([0, 1], [f"Task {task}", f"Task {task + 1}"])
    plt.yticks([0, 1], [f"Task {task}", f"Task {task + 1}"])
    plt.xlabel("Predicted class", fontsize=16)
    plt.ylabel("True class", fontsize=16)
    fig = plt.gcf()
    ax = fig.axes[0]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="k", fontsize=16)
    ax.tick_params(labelsize=16)
    fig.set_size_inches(8, 8)
    plt.savefig(f"{data_dir}/confusion_matrix_task_{task}_{task + 1}.pdf")
    plt.close()

    print(f"threshold={threshold}")
    print(f"Normalized confusion matrix (rows=true class, cols=predicted class), "
          f"task {task} (label 0, n={X_own_train.shape[0]}) vs "
          f"task {task + 1} (label 1, n={X_next.shape[0]}):")
    print(matrix)
    return matrix


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--task", type=int, required=True,
                         help="Source task index (0 or 1) whose memory unit to train/evaluate.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-train", type=int, default=25000)
    parser.add_argument("--n-eval", type=int, default=1000,
                         help="[evaluate] number of task+1 examples used as the anomaly class.")
    parser.add_argument("--epochs", type=int, default=200, help="[train]")
    parser.add_argument("--batch-size", type=int, default=128, help="[train]")
    parser.add_argument("--threshold", type=float, default=MEMORY_MSE_THRESHOLD, help="[evaluate]")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode == "train":
        train_memory_unit(args.task, data_dir=args.data_dir, n_train=args.n_train,
                           epochs=args.epochs, batch_size=args.batch_size)
    else:
        evaluate_memory_unit(args.task, data_dir=args.data_dir, threshold=args.threshold,
                              n_train=args.n_train, n_eval=args.n_eval)
