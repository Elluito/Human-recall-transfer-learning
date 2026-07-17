# layouts/

`pacman.py`'s `crear_layout(difficulty)` procedurally (re)writes
`campo_<difficulty>.lay` here at the start of every run and before every
episode, placing the food, Pacman, the ghost, and the walls that define each
task's difficulty (see Figure 3 in the paper). You don't need to create or
edit anything here; the directory just needs to exist and be writable.
