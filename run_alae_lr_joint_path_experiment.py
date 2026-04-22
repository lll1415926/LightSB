"""
Joint path training for a discrete two-stage Low-Rank LightSB bridge.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALAE"))

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from alae_ffhq_inference import decode, load_model
from lightsb_lr_diag import LowRankDiagLightSBPotential
from src.distributions import TensorSampler


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
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--n-potentials", type=int, default=12)
    p.add_argument("--stage1-rank", type=int, default=4)
    p.add_argument("--stage1-epsilon", type=float, default=0.5)
    p.add_argument("--stage2-rank", type=int, default=6)
    p.add_argument("--stage2-epsilon", type=float, default=0.4)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--init-delta", type=float, default=-4.0)
    p.add_argument("--init-mean-scale", type=float, default=0.2)
    p.add_argument("--init-u-std", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--d-lr-main", type=float, default=5e-4)
    p.add_argument("--d-lr-shape", type=float, default=1e-4)
    p.add_argument("--grad-max-norm", type=float, default=5.0)
    p.add_argument("--lambda-diag", type=float, default=0.0)
    p.add_argument("--lambda-U", dest="lambda_U", type=float, default=1e-4)
    p.add_argument("--lambda-m", type=float, default=0.0)
    p.add_argument("--lambda-path", type=float, default=1.0)
    p.add_argument("--lambda-mid-mean", type=float, default=10.0)
    p.add_argument("--lambda-mid-var", type=float, default=2.0)
    p.add_argument("--lambda-stage1-anchor", type=float, default=5.0)
    p.add_argument("--path-warmup-steps", type=int, default=2000)
    p.add_argument("--stage1-lr-scale", type=float, default=0.25)
    p.add_argument("--stage2-lr-scale", type=float, default=1.0)
    p.add_argument("--load-pretrained", action="store_true")
    p.add_argument("--no-load-pretrained", dest="load_pretrained", action="store_false")
    p.set_defaults(load_pretrained=True)
    p.add_argument("--output-seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--print-every", type=int, default=200)
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


def conditional_mean(model, x):
    params = model.compute_component_params(x)
    weights = torch.softmax(params["log_w"], dim=-1)
    return (weights.unsqueeze(-1) * params["mu"]).sum(dim=1)


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


def stage_exp_name(src, tgt, epsilon, rank, suffix=""):
    name = f"LightSB_LR_ALAE_{src}_TO_{tgt}_EPSILON_{epsilon}_RANK_{rank}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


exp_name = (
    f"LightSB_LR_ALAE_JOINT_{args.input_data}_TO_{args.middle_data}_TO_{args.target_data}"
    f"_S1E_{args.stage1_epsilon}_R{args.stage1_rank}"
    f"_S2E_{args.stage2_epsilon}_R{args.stage2_rank}"
)
if args.exp_suffix.strip():
    exp_name = f"{exp_name}_{args.exp_suffix.strip()}"
output_dir = os.path.join(ROOT, "checkpoints", exp_name)
os.makedirs(output_dir, exist_ok=True)

print(f"=== {exp_name} ===")
print(f"Output: {output_dir}")

latents = np.load(os.path.join(ROOT, "data", "latents.npy"))
gender = np.load(os.path.join(ROOT, "data", "gender.npy"))
age = np.load(os.path.join(ROOT, "data", "age.npy"))
test_inp_imgs = np.load(os.path.join(ROOT, "data", "test_images.npy"))

train_latents, test_latents = latents[:TRAIN_SIZE], latents[TRAIN_SIZE:]
train_gender, test_gender = gender[:TRAIN_SIZE], gender[TRAIN_SIZE:]
train_age, test_age = age[:TRAIN_SIZE], age[TRAIN_SIZE:]

x0_inds_train = get_inds(train_gender, train_age, args.input_data)
x1_inds_train = get_inds(train_gender, train_age, args.middle_data)
x2_inds_train = get_inds(train_gender, train_age, args.target_data)
x0_inds_test = get_inds(test_gender, test_age, args.input_data)

X0_train = torch.tensor(train_latents[x0_inds_train], dtype=torch.float32)
X1_train = torch.tensor(train_latents[x1_inds_train], dtype=torch.float32)
X2_train = torch.tensor(train_latents[x2_inds_train], dtype=torch.float32)

print(
    f"Train sizes: {args.input_data}={len(X0_train)}, "
    f"{args.middle_data}={len(X1_train)}, "
    f"{args.target_data}={len(X2_train)}"
)
print(
    f"Age bins: young=[{args.young_min_age}, {args.young_max_age}), "
    f"old=[{args.old_min_age}, {args.old_max_age})"
)

assert len(X0_train) > 0
assert len(X1_train) > 0
assert len(X2_train) > 0
assert len(x0_inds_test) > 0

X0_sampler = TensorSampler(X0_train, device="cpu")
X1_sampler = TensorSampler(X1_train, device="cpu")
X2_sampler = TensorSampler(X2_train, device="cpu")

torch.manual_seed(args.output_seed)
np.random.seed(args.output_seed)

D1 = build_model(args.stage1_rank, args.stage1_epsilon)
D2 = build_model(args.stage2_rank, args.stage2_epsilon)
D1_ref = build_model(args.stage1_rank, args.stage1_epsilon)
D1_ref.requires_grad_(False)
D1_ref.eval()

with torch.no_grad():
    D1.initialize_means_from_samples(X1_sampler.sample(args.n_potentials))
    D2.initialize_means_from_samples(X2_sampler.sample(args.n_potentials))
    if getattr(D1, "r", 0) > 0:
        D1.U.normal_(mean=0.0, std=args.init_u_std)
    if getattr(D2, "r", 0) > 0:
        D2.U.normal_(mean=0.0, std=args.init_u_std)
    D1.log_alpha_raw.zero_()
    D2.log_alpha_raw.zero_()
    D1.recenter_alpha()
    D2.recenter_alpha()

if args.load_pretrained:
    stage1_ckpt = os.path.join(
        ROOT,
        "checkpoints",
        stage_exp_name(
            args.input_data,
            args.middle_data,
            args.stage1_epsilon,
            args.stage1_rank,
        ),
        "D.pt",
    )
    stage2_ckpt = os.path.join(
        ROOT,
        "checkpoints",
        stage_exp_name(
            args.middle_data,
            args.target_data,
            args.stage2_epsilon,
            args.stage2_rank,
            "stage2",
        ),
        "D.pt",
    )
    if os.path.exists(stage1_ckpt):
        print(f"Loading pretrained stage1: {stage1_ckpt}")
        state = torch.load(stage1_ckpt, map_location="cpu")
        D1.load_state_dict(state)
        D1_ref.load_state_dict(state)
    else:
        print(f"Pretrained stage1 not found: {stage1_ckpt}")
        D1_ref.load_state_dict(D1.state_dict())

    if os.path.exists(stage2_ckpt):
        print(f"Loading pretrained stage2: {stage2_ckpt}")
        D2.load_state_dict(torch.load(stage2_ckpt, map_location="cpu"))
    else:
        print(f"Pretrained stage2 not found: {stage2_ckpt}")
else:
    D1_ref.load_state_dict(D1.state_dict())

stage1_main_params = [D1.log_alpha_raw, D1.m]
stage1_shape_params = [D1.delta]
if getattr(D1, "r", 0) > 0:
    stage1_shape_params.append(D1.U)

stage2_main_params = [D2.log_alpha_raw, D2.m]
stage2_shape_params = [D2.delta]
if getattr(D2, "r", 0) > 0:
    stage2_shape_params.append(D2.U)

optimizer = torch.optim.Adam(
    [
        {
            "params": stage1_main_params,
            "lr": args.d_lr_main * args.stage1_lr_scale,
        },
        {
            "params": stage1_shape_params,
            "lr": args.d_lr_shape * args.stage1_lr_scale,
        },
        {
            "params": stage2_main_params,
            "lr": args.d_lr_main * args.stage2_lr_scale,
        },
        {
            "params": stage2_shape_params,
            "lr": args.d_lr_shape * args.stage2_lr_scale,
        },
    ]
)

ckpt1 = os.path.join(output_dir, "D1.pt")
ckpt2 = os.path.join(output_dir, "D2.pt")
opt_ckpt = os.path.join(output_dir, "joint_opt.pt")

if os.path.exists(ckpt1) and os.path.exists(ckpt2):
    print("Found joint checkpoints, loading and skipping training.")
    D1.load_state_dict(torch.load(ckpt1, map_location="cpu"))
    D2.load_state_dict(torch.load(ckpt2, map_location="cpu"))
else:
    history = {
        "loss": [],
        "local1": [],
        "local2": [],
        "path": [],
        "mid_mean": [],
        "mid_var": [],
        "stage1_anchor": [],
        "reg": [],
    }

    for step in tqdm(range(args.max_steps)):
        optimizer.zero_grad()

        x0 = X0_sampler.sample(args.batch_size)
        x1 = X1_sampler.sample(args.batch_size)
        x2 = X2_sampler.sample(args.batch_size)

        local1 = D1.training_loss(x0, x1)
        local2 = D2.training_loss(x1, x2)

        x1_hat = conditional_mean(D1, x0)
        path = D2.training_loss(x1_hat, x2)
        x1_ref = conditional_mean(D1_ref, x0)

        x1_mean = x1.mean(dim=0)
        x1_hat_mean = x1_hat.mean(dim=0)
        mid_mean = (x1_hat_mean - x1_mean).pow(2).mean()

        x1_var = x1.var(dim=0, unbiased=False)
        x1_hat_var = x1_hat.var(dim=0, unbiased=False)
        mid_var = (x1_hat_var - x1_var).pow(2).mean()
        stage1_anchor = (x1_hat - x1_ref).pow(2).mean()

        if args.path_warmup_steps > 0:
            warmup = min(1.0, float(step + 1) / float(args.path_warmup_steps))
        else:
            warmup = 1.0

        reg = D1.regularization(
            lambda_diag=args.lambda_diag,
            lambda_U=args.lambda_U,
            lambda_m=args.lambda_m,
        ) + D2.regularization(
            lambda_diag=args.lambda_diag,
            lambda_U=args.lambda_U,
            lambda_m=args.lambda_m,
        )

        loss = (
            local1
            + local2
            + warmup * args.lambda_path * path
            + warmup * args.lambda_mid_mean * mid_mean
            + warmup * args.lambda_mid_var * mid_var
            + args.lambda_stage1_anchor * stage1_anchor
            + reg
        )
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(D1.parameters()) + list(D2.parameters()),
            args.grad_max_norm,
        )
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["local1"].append(float(local1.item()))
        history["local2"].append(float(local2.item()))
        history["path"].append(float(path.item()))
        history["mid_mean"].append(float(mid_mean.item()))
        history["mid_var"].append(float(mid_var.item()))
        history["stage1_anchor"].append(float(stage1_anchor.item()))
        history["reg"].append(float(reg.item()))

        if step % args.print_every == 0 or step == args.max_steps - 1:
            print(
                f"[step {step:5d}] "
                f"loss={loss.item():+.6f} | "
                f"l1={local1.item():+.6f} | "
                f"l2={local2.item():+.6f} | "
                f"path={path.item():+.6f} | "
                f"mid_mean={mid_mean.item():.6e} | "
                f"mid_var={mid_var.item():.6e} | "
                f"s1_anchor={stage1_anchor.item():.6e} | "
                f"warmup={warmup:.3f} | "
                f"reg={reg.item():.6e} | "
                f"grad={grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm):.6e}"
            )

    torch.save(D1.state_dict(), ckpt1)
    torch.save(D2.state_dict(), ckpt2)
    torch.save(optimizer.state_dict(), opt_ckpt)

    for key, values in history.items():
        arr = np.array(values)
        np.save(os.path.join(output_dir, f"{key}.npy"), arr)
        plt.figure(figsize=(6, 3), dpi=150)
        plt.plot(arr)
        plt.title(key)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{key}.png"))
        plt.close()

alae_cfg = os.path.join(ROOT, "ALAE", "configs", "ffhq.yaml")
alae_dir = os.path.join(ROOT, "ALAE", "training_artifacts", "ffhq")
os.chdir(os.path.join(ROOT, "ALAE"))
alae_model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
os.chdir(ROOT)

try:
    alae_device = next(alae_model.parameters()).device
except StopIteration:
    alae_device = torch.device("cpu")

np.random.seed(args.output_seed)
valid_pos = np.where(x0_inds_test < len(test_inp_imgs))[0]
n_show = min(args.n_show, len(valid_pos))
inds_to_map = np.random.choice(valid_pos, size=n_show, replace=False)
lat_idx = x0_inds_test[inds_to_map]

latent_to_map = torch.tensor(test_latents[lat_idx], dtype=torch.float32)
inp_images = test_inp_imgs[lat_idx]

with torch.no_grad():
    stage1_mid = conditional_mean(D1, latent_to_map)
    stage2_final = D2.conditional_sample(stage1_mid, n_samples=args.n_final_samples)

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
axes[0][1].set_title("Stage1 mean", fontsize=5)
for k in range(args.n_final_samples):
    axes[0][k + 2].set_title(f"Final {k + 1}", fontsize=5)

fig.suptitle(
    f"Joint {args.input_data} -> {args.middle_data} -> {args.target_data}",
    fontsize=7,
    y=1.01,
)
fig.tight_layout(pad=0.05)

out_fig = os.path.join(output_dir, "result.png")
fig.savefig(out_fig, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_fig}")
