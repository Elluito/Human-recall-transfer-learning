# models/

Trained Q-network checkpoints (`.h5`) go here. They are not committed to the
repository (each is tens of MB and there is one per training run).

To run a `--transfer` experiment you need the *source* task's checkpoint
here, named exactly as `qlearningAgents.MODEL_REGISTRY` expects:

| Task | Expected filename | Download |
|---|---|---|
| 0 | `modelo_imagen_20000_04_01_dif0_1575607728_gamma_0.9_attemp_8.h5` | [Google Drive](https://drive.google.com/file/d/14ObAYHNWIO9jS1yfnY1-CiyqgzosmCV2/view?usp=sharing) |
| 1 | `modelo_imagen_25000_04_01_dif1_1576737275_attemp_3_gamma0.9.h5` | [Google Drive](https://drive.google.com/file/d/1QifvqTpnngVA6eyU147BaSQS6jp4SKHo/view?usp=sharing) |
| 2 | `modelo_imagen_40000_09_01_dif_2_attemp_4_gamma0.9_transfer_from_1.h5` | produced by your own T1->T2 training run (see top-level README) |

You can train task 0 and task 1 from scratch yourself instead of downloading
them (see "Reproducing the experiments" in the top-level README); the
downloads just save the ~25-40k training episodes that would otherwise be
needed before you can run a transfer experiment.

If you save a checkpoint under a different name, either rename the file to
match the table above or edit `MODEL_REGISTRY` at the top of
`../qlearningAgents.py`.
