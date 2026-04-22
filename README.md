# Light Schrodinger Bridge

This repository contains `PyTorch` code for reproducing experiments from
**Light Schrodinger Bridge** ([arXiv](https://arxiv.org/abs/2310.01174),
[OpenReview](https://openreview.net/forum?id=WhZoCLRWYJ)) by
[Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en),
[Nikita Gushchin](https://scholar.google.com/citations?user=UaRTbNoAAAAJ&hl=en&oi=ao),
and
[Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru).

One representative example is unpaired `Male -> Female` image translation in
the latent space of ALAE for `1024 x 1024` FFHQ images.

<p align="center"><img src="teaser/teaser.png" width="800" /></p>

## Repository overview

Most experiments are provided as notebooks in `notebooks/`, while reusable
implementation code lives in `src/` and `ALAE/`.

Main directories:

- `ALAE/` - ALAE model code and utilities.
- `src/` - core LightSB modules and supporting components.
- `checkpoints/` - saved outputs and experiment artifacts.
- `notebooks/` - experiment notebooks.

Notebook entry points:

- `notebooks/LightSB_swiss_roll.ipynb` - swiss roll experiments.
- `notebooks/swiss_roll_plot.ipynb` - plotting for swiss roll results.
- `notebooks/LightSB_EOT_benchmark.ipynb` - EOT benchmark experiments.
- `notebooks/LightSB_single_cell.ipynb` - single-cell experiments.
- `notebooks/LightSB_alae.ipynb` - image experiments with ALAE.
- `notebooks/LightSB_MSCI.ipynb` - MSCI-related experiments.

This repository uses `wandb` for experiment tracking in several scripts and
notebooks.

## Image experiments

For image translation in ALAE latent space, the main top-level scripts are:

- `run_alae_experiment.py` - baseline ALAE latent LightSB experiment.
- `run_alae_lr_experiment.py` - low-rank variant for ALAE latent transport.
- `run_full_alae_training.py` - longer ALAE latent training run with exposed
  training controls.
- `lightsb_lr_diag.py` - low-rank diagonal potential parameterization used by
  the low-rank experiment scripts.

## Parent-to-children experiment

Besides image translation, this repository also contains a
**parent-to-children latent transport** setup. Here the source is a pair of
parent latents `(z_f, z_m)` and the target is a child latent `y`.

Main entry point:

- `run_parent_child_static_experiment.py` - trains a conditional static
  Schrodinger / entropic-OT model from joint parent latents to child latents.

Supporting implementation:

- `src/conditional_static_lightsb.py` - static LightSB-style conditional model
  for `(father latent, mother latent) -> child latent`.

Supported inputs:

1. A user-provided `.npz` dataset containing `z_f`, `z_m`, `y`, and optionally
   `a` and `split`.
2. A synthetic dataset via `--synthetic` for quick smoke tests.

Expected `.npz` arrays:

- `z_f`: father latents with shape `[N, d]`
- `z_m`: mother latents with shape `[N, d]`
- `y`: child latents with shape `[N, d]`
- `a` (optional): integer condition labels with shape `[N]`
- `split` (optional): train/test split labels

Example commands:

```bash
python run_parent_child_static_experiment.py --synthetic --latent-dim 32
python run_parent_child_static_experiment.py --data-path path/to/parent_child_latents.npz --latent-dim 32
```

## Discrete multi-stage view

For image translation, multi-stage LightSB can be interpreted as a
**discrete Schrodinger bridge** built on a sequence of intermediate marginals
instead of a single bridge between source and target endpoints.

In the one-stage setting, we learn a conditional bridge from `p_0` to `p_1`.
In the discrete multi-stage setting, we introduce intermediate marginals
`p_0, p_1, ..., p_T` and learn a sequence of local bridges
`pi_t(x_t | x_{t-1})` for `t = 1, ..., T`, giving the factorization

`p(x_0, ..., x_T) = p_0(x_0) * pi_1(x_1 | x_0) * ... * pi_T(x_T | x_{T-1})`.

This is useful when the endpoint shift is too large for a single LightSB map.
For example, instead of solving `YOUNG_MAN -> OLD_WOMAN` in one step, we may
use the discrete path

`YOUNG_MAN -> OLD_MAN -> OLD_WOMAN`.

Here, the first bridge mainly changes age, while the second mainly changes
gender. Each local transport problem is easier because consecutive marginals are
closer than the original endpoints.

Two-stage experiment drivers:

- `run_alae_standard_two_stage_experiment.py` - standard LightSB two-stage
  ALAE experiments.
- `run_alae_lr_two_stage_experiment.py` - low-rank LightSB two-stage ALAE
  experiments.
- `run_alae_lr_joint_path_experiment.py` - joint training for a discrete
  two-stage low-rank path with local bridge losses, a path-coupling loss, and
  intermediate moment-matching terms.

The main caveat is error accumulation: if an early stage drifts away from the
intended intermediate marginal, later stages operate on an already shifted
distribution.

## Citation

```text
@inproceedings{
korotin2024light,
title={Light Schr\"odinger Bridge},
author={Alexander Korotin and Nikita Gushchin and Evgeny Burnaev},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=WhZoCLRWYJ}
}
```
