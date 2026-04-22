"""
Two-stage Low-Rank LightSB ALAE image translation experiment.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALAE"))

import numpy as np
import torch
from matplotlib import pyplot as plt

from alae_ffhq_inference import decode, load_model
from lightsb_lr_diag import LowRankDiagLightSBPotential


ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SIZE = 60000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-data", type=str, default="YOUNG_MAN")
    p.add_argument("--middle-data", type=str, default="OLD_MAN")
    p.add_argument("--target-data", type=str, default="OLD_WOMAN")
    p.add_argument("--young-min-age", type=int, default=18)
    p.add_argument("--young-max-age", type=int, default=30)
    p.add_argument("--old-min-age", type=int, default=55)
    p.add_argument("--old-max-age", type=int, default=100)
    p.add_argument("--n-potentials", type=int, default=12)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--stage1-rank", type=int, default=4)
    p.add_argument("--stage1-epsilon", type=float, default=0.5)
    p.add_argument("--stage1-suffix", type=str, default="")
    p.add_argument("--stage2-rank", type=int, default=6)
    p.add_argument("--stage2-epsilon", type=float, default=0.4)
    p.add_argument("--stage2-suffix", type=str, default="stage2")
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--init-delta", type=float, default=-4.0)
    p.add_argument("--init-mean-scale", type=float, default=0.2)
    p.add_argument("--output-seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--n-show", type=int, default=8)
    p.add_argument("--n-final-samples", type=int, default=3)
    p.add_argument("--exp-suffix", type=str, default="")
    return p.parse_args()


args = parse_args()


def get_inds(g, a, label):
    g = g.reshape(-1)
    a = a.reshape(-1)
    valid_age = (a != -1)
    male = (g == "male")
    female = (g == "female")

    if label == "MAN":
        return np.where(male)[0]
    if label == "WOMAN":
        return np.where(female)[0]
    if label == "ADULT":
        return np.where((a >= 18) & valid_age)[0]
    if label == "CHILDREN":
        return np.where((a < 18) & valid_age)[0]
    if label == "BOY":
        return np.where(male & valid_age & (a < 18))[0]
    if label == "GIRL":
        return np.where(female & valid_age & (a < 18))[0]
    if label == "YOUNG":
        return np.where(
            valid_age & (a >= args.young_min_age) & (a < args.young_max_age)
        )[0]
    if label == "OLD":
        return np.where(
            valid_age & (a >= args.old_min_age) & (a < args.old_max_age)
        )[0]
    if label == "YOUNG_MAN":
        return np.where(
            male & valid_age & (a >= args.young_min_age) & (a < args.young_max_age)
        )[0]
    if label == "OLD_MAN":
        return np.where(
            male & valid_age & (a >= args.old_min_age) & (a < args.old_max_age)
        )[0]
    if label == "YOUNG_WOMAN":
        return np.where(
            female & valid_age & (a >= args.young_min_age) & (a < args.young_max_age)
        )[0]
    if label == "OLD_WOMAN":
        return np.where(
            female & valid_age & (a >= args.old_min_age) & (a < args.old_max_age)
        )[0]
    raise ValueError(f"Unknown label: {label}")


def exp_name(src, tgt, epsilon, rank, suffix):
    name = f"LightSB_LR_ALAE_{src}_TO_{tgt}_EPSILON_{epsilon}_RANK_{rank}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


def build_model(rank, epsilon):
    return LowRankDiagLightSBPotential(
        d=args.dim,
        K=args.n_potentials,
        r=rank,
        sigma=args.sigma,
        eps=args.eps,
        init_delta=args.init_delta,
        init_mean_scale=args.init_mean_scale,
        alpha_scale=epsilon,
    ).cpu()


def decode_batch(model, z):
    with torch.no_grad():
        img = decode(model, z)
        return (
            ((img * 0.5 + 0.5) * 255)
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )


stage1_name = exp_name(
    args.input_data, args.middle_data, args.stage1_epsilon, args.stage1_rank, args.stage1_suffix
)
stage2_name = exp_name(
    args.middle_data, args.target_data, args.stage2_epsilon, args.stage2_rank, args.stage2_suffix
)
two_stage_name = (
    f"LightSB_LR_ALAE_{args.input_data}_TO_{args.middle_data}_TO_{args.target_data}"
    f"_S1E_{args.stage1_epsilon}_R{args.stage1_rank}"
    f"_S2E_{args.stage2_epsilon}_R{args.stage2_rank}"
)
if args.exp_suffix.strip():
    two_stage_name = f"{two_stage_name}_{args.exp_suffix.strip()}"

output_dir = os.path.join(ROOT, "checkpoints", two_stage_name)
os.makedirs(output_dir, exist_ok=True)

print(f"=== {two_stage_name} ===")
print(f"Output: {output_dir}")
print(f"Stage 1 checkpoint: {stage1_name}")
print(f"Stage 2 checkpoint: {stage2_name}")

latents = np.load(os.path.join(ROOT, "data", "latents.npy"))
gender = np.load(os.path.join(ROOT, "data", "gender.npy"))
age = np.load(os.path.join(ROOT, "data", "age.npy"))
test_inp_imgs = np.load(os.path.join(ROOT, "data", "test_images.npy"))

test_latents = latents[TRAIN_SIZE:]
test_gender = gender[TRAIN_SIZE:]
test_age = age[TRAIN_SIZE:]

x_inds_test = get_inds(test_gender, test_age, args.input_data)
valid_pos = np.where(x_inds_test < len(test_inp_imgs))[0]
assert len(valid_pos) > 0, "No valid source test images found."

torch.manual_seed(args.output_seed)
np.random.seed(args.output_seed)

n_show = min(args.n_show, len(valid_pos))
inds_to_map = np.random.choice(valid_pos, size=n_show, replace=False)
lat_idx = x_inds_test[inds_to_map]
latent_to_map = torch.tensor(test_latents[lat_idx], dtype=torch.float32)
inp_images = test_inp_imgs[lat_idx]

stage1 = build_model(args.stage1_rank, args.stage1_epsilon)
stage1_ckpt = os.path.join(ROOT, "checkpoints", stage1_name, "D.pt")
assert os.path.exists(stage1_ckpt), f"Missing stage 1 checkpoint: {stage1_ckpt}"
stage1.load_state_dict(torch.load(stage1_ckpt, map_location="cpu"))
stage1.eval()

stage2 = build_model(args.stage2_rank, args.stage2_epsilon)
stage2_ckpt = os.path.join(ROOT, "checkpoints", stage2_name, "D.pt")
assert os.path.exists(stage2_ckpt), f"Missing stage 2 checkpoint: {stage2_ckpt}"
stage2.load_state_dict(torch.load(stage2_ckpt, map_location="cpu"))
stage2.eval()

with torch.no_grad():
    stage1_mid = stage1.conditional_sample(latent_to_map, n_samples=1)[:, 0, :]
    stage2_final = stage2.conditional_sample(
        stage1_mid, n_samples=args.n_final_samples
    )

alae_cfg = os.path.join(ROOT, "ALAE", "configs", "ffhq.yaml")
alae_dir = os.path.join(ROOT, "ALAE", "training_artifacts", "ffhq")
os.chdir(os.path.join(ROOT, "ALAE"))
alae_model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
os.chdir(ROOT)

try:
    alae_device = next(alae_model.parameters()).device
except StopIteration:
    alae_device = torch.device("cpu")

stage1_mid = stage1_mid.to(alae_device)
stage2_final = stage2_final.to(alae_device)

mid_imgs = decode_batch(alae_model, stage1_mid)
final_imgs = []
for k in range(args.n_final_samples):
    final_imgs.append(decode_batch(alae_model, stage2_final[:, k]))
final_imgs = np.stack(final_imgs, axis=1)

fig, axes = plt.subplots(
    n_show,
    args.n_final_samples + 2,
    figsize=(args.n_final_samples + 2, n_show),
    dpi=200,
)
for i in range(n_show):
    axes[i][0].imshow(inp_images[i])
    axes[i][0].axis("off")
    axes[i][1].imshow(mid_imgs[i])
    axes[i][1].axis("off")
    for k in range(args.n_final_samples):
        axes[i][k + 2].imshow(final_imgs[i, k])
        axes[i][k + 2].axis("off")

axes[0][0].set_title("Input", fontsize=5)
axes[0][1].set_title("Stage1", fontsize=5)
for k in range(args.n_final_samples):
    axes[0][k + 2].set_title(f"Final {k + 1}", fontsize=5)

fig.suptitle(
    f"{args.input_data} -> {args.middle_data} -> {args.target_data}",
    fontsize=7,
    y=1.01,
)
fig.tight_layout(pad=0.05)

out_fig = os.path.join(output_dir, "result.png")
fig.savefig(out_fig, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_fig}")
