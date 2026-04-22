# Light Schrödinger Bridge

This repository contains the `PyTorch` code to reproduce the experiments from work **Light Schrödinger Bridge** (LightSB paper on [arxiv](https://arxiv.org/abs/2310.01174) and [OpenReview](https://openreview.net/forum?id=WhZoCLRWYJ)) by  [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Nikita Gushchin](https://scholar.google.com/citations?user=UaRTbNoAAAAJ&hl=en&oi=ao) and [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru).

**An example:** Unpaired *Male* -> *Female* translation by our LightSB solver applied in the latent space of ALAE for 1024x1024 FFHQ images. *Our LightSB converges on 4 cpu cores in less than 1 minute.*

<p align="center"><img src="teaser/teaser.png" width="800" /></p>

## Repository structure:
All the experiments are issued in the form of pretty self-explanatory jupyter notebooks (`notebooks/`). Auxilary source code is moved to `.py` modules (`src/`). 

Note that we use `wandb` ([link](https://wandb.ai/site)) dashboard system when launching our experiments. The practitioners are expected to use `wandb` too. 

```notebooks/Toy_experiments.ipynb``` - Toy experiments.

```ALAE``` - Code for the ALAE model.

```src``` - LightSB implementation and axuliary code for plotting.

```notebooks/LightSB_swiss_roll.ipynb``` - Code for swiss roll experiments.

```notebooks/swiss_roll_plot.ipynb``` - Code for plotting the reported image for swiss roll.

```notebooks/LightSB_EOT_benchmark.ipynb``` - Code for benchmark experiments.

```notebooks/LightSB_single_cell.ipynb``` - Code for single cell experiments.

```notebooks/LightSB_alae.ipynb``` - Code for image experiments with ALAE.

## Discrete multi-stage view
For the image translation setup, we can interpret multi-stage LightSB as a
**discrete Schrödinger bridge** built on a sequence of intermediate marginals,
instead of a single bridge between the source and target endpoints.

In the one-stage setting, we learn a conditional bridge from a source marginal
`p_0` to a target marginal `p_1`. In the discrete multi-stage setting, we
introduce intermediate marginals
`p_0, p_1, ..., p_T` and learn a sequence of local bridges
`pi_t(x_t | x_{t-1})` for `t = 1, ..., T`. The resulting path measure factorizes as

`p(x_0, ..., x_T) = p_0(x_0) * pi_1(x_1 | x_0) * ... * pi_T(x_T | x_{T-1}).`

This viewpoint is useful when the endpoint shift is too large for a single
LightSB map. For example, instead of solving
`YOUNG_MAN -> OLD_WOMAN` in one step, we may use the discrete path

`YOUNG_MAN -> OLD_MAN -> OLD_WOMAN`.

Here, the first bridge mainly changes age, while the second bridge mainly
changes gender. Each local transport problem is simpler because consecutive
marginals are closer than the original endpoints. In practice, this can reduce
collapse toward an average target face and make the bridge easier to optimize.

This repository includes simple two-stage experiment drivers for that discrete
view:

`run_alae_standard_two_stage_experiment.py` - Standard LightSB two-stage ALAE experiments.

`run_alae_lr_two_stage_experiment.py` - Low-rank LightSB two-stage ALAE experiments.

`run_alae_lr_joint_path_experiment.py` - Joint training for a discrete two-stage
low-rank path with local bridge losses, a path-coupling loss, and intermediate
moment-matching terms.

The main caveat is that multi-stage transport accumulates error: if an early
stage drifts away from the intended intermediate marginal, later stages operate
on an already shifted distribution. So the quality of the full discrete bridge
is bottlenecked by the weakest local stage.

## Citation
```
@inproceedings{
korotin2024light,
title={Light Schr\"odinger Bridge},
author={Alexander Korotin and Nikita Gushchin and Evgeny Burnaev},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=WhZoCLRWYJ}
}
```
