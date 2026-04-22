"""
Low-Rank Diagonal LightSB ALAE Image Translation Experiment.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALAE"))

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

from alae_ffhq_inference import decode, load_model
from lightsb_lr_diag import LowRankDiagLightSBPotential
from src.distributions import TensorSampler


ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-data", type=str, default="YOUNG_MAN")
    p.add_argument("--target-data", type=str, default="OLD_MAN")
    p.add_argument("--young-min-age", type=int, default=18)
    p.add_argument("--young-max-age", type=int, default=30)
    p.add_argument("--old-min-age", type=int, default=55)
    p.add_argument("--old-max-age", type=int, default=100)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--n-potentials", type=int, default=12)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--init-delta", type=float, default=-4.0)
    p.add_argument("--init-mean-scale", type=float, default=0.2)
    p.add_argument("--init-u-std", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--d-lr-main", type=float, default=5e-4)
    p.add_argument("--d-lr-shape", type=float, default=1e-4)
    p.add_argument("--grad-max-norm", type=float, default=5.0)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--output-seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--lambda-diag", type=float, default=0.0)
    p.add_argument("--lambda-U", dest="lambda_U", type=float, default=1e-4)
    p.add_argument("--lambda-m", type=float, default=0.0)
    p.add_argument("--exp-suffix", type=str, default="")
    return p.parse_args()


args = parse_args()

INPUT_DATA = args.input_data
TARGET_DATA = args.target_data
YOUNG_MIN_AGE = args.young_min_age
YOUNG_MAX_AGE = args.young_max_age
OLD_MIN_AGE = args.old_min_age
OLD_MAX_AGE = args.old_max_age
DIM = args.dim
EPSILON = args.epsilon
N_POTENTIALS = args.n_potentials
RANK = args.rank
SIGMA = args.sigma
EPS = args.eps
INIT_DELTA = args.init_delta
INIT_MEAN_SCALE = args.init_mean_scale
INIT_U_STD = args.init_u_std
BATCH_SIZE = args.batch_size
D_LR_MAIN = args.d_lr_main
D_LR_SHAPE = args.d_lr_shape
D_GRADIENT_MAX_NORM = args.grad_max_norm
MAX_STEPS = args.max_steps
OUTPUT_SEED = args.output_seed
PRINT_EVERY = args.print_every
LAMBDA_DIAG = args.lambda_diag
LAMBDA_U = args.lambda_U
LAMBDA_M = args.lambda_m
EXP_SUFFIX = args.exp_suffix.strip()

EXP_NAME = (
    f"LightSB_LR_ALAE_{INPUT_DATA}_TO_{TARGET_DATA}"
    f"_EPSILON_{EPSILON}_RANK_{RANK}"
)
if EXP_SUFFIX:
    EXP_NAME = f"{EXP_NAME}_{EXP_SUFFIX}"
OUTPUT_PATH = os.path.join(ROOT, "checkpoints", EXP_NAME)
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"=== {EXP_NAME} ===")
print(f"Output: {OUTPUT_PATH}")

print("\n[1/4] Loading data...")
data_dir = os.path.join(ROOT, "data")
train_size, test_size = 60000, 10000

latents = np.load(os.path.join(data_dir, "latents.npy"))
gender = np.load(os.path.join(data_dir, "gender.npy"))
age = np.load(os.path.join(data_dir, "age.npy"))
test_inp_imgs = np.load(os.path.join(data_dir, "test_images.npy"))

train_latents, test_latents = latents[:train_size], latents[train_size:]
train_gender, test_gender = gender[:train_size], gender[train_size:]
train_age, test_age = age[:train_size], age[train_size:]


def get_inds(g, a, split, label):
    g = g.reshape(-1)
    a = a.reshape(-1)

    valid_age = (a != -1)
    male = (g == "male")
    female = (g == "female")

    if label == "MAN":
        return np.where(male)[0]
    elif label == "WOMAN":
        return np.where(female)[0]
    elif label == "ADULT":
        return np.where((a >= 18) & valid_age)[0]
    elif label == "CHILDREN":
        return np.where((a < 18) & valid_age)[0]
    elif label == "BOY":
        return np.where(male & valid_age & (a < 18))[0]
    elif label == "GIRL":
        return np.where(female & valid_age & (a < 18))[0]
    elif label == "YOUNG":
        return np.where(valid_age & (a >= YOUNG_MIN_AGE) & (a < YOUNG_MAX_AGE))[0]
    elif label == "OLD":
        return np.where(valid_age & (a >= OLD_MIN_AGE) & (a < OLD_MAX_AGE))[0]
    elif label == "YOUNG_MAN":
        return np.where(
            male & valid_age & (a >= YOUNG_MIN_AGE) & (a < YOUNG_MAX_AGE)
        )[0]
    elif label == "OLD_MAN":
        return np.where(
            male & valid_age & (a >= OLD_MIN_AGE) & (a < OLD_MAX_AGE)
        )[0]
    elif label == "YOUNG_WOMAN":
        return np.where(
            female & valid_age & (a >= YOUNG_MIN_AGE) & (a < YOUNG_MAX_AGE)
        )[0]
    elif label == "OLD_WOMAN":
        return np.where(
            female & valid_age & (a >= OLD_MIN_AGE) & (a < OLD_MAX_AGE)
        )[0]
    else:
        raise ValueError(f"Unknown label: {label}")


x_inds_train = get_inds(train_gender, train_age, "train", INPUT_DATA)
x_inds_test = get_inds(test_gender, test_age, "test", INPUT_DATA)
y_inds_train = get_inds(train_gender, train_age, "train", TARGET_DATA)
y_inds_test = get_inds(test_gender, test_age, "test", TARGET_DATA)

X_train = torch.tensor(train_latents[x_inds_train], dtype=torch.float32)
Y_train = torch.tensor(train_latents[y_inds_train], dtype=torch.float32)
X_test = torch.tensor(test_latents[x_inds_test], dtype=torch.float32)

print(f"  Source ({INPUT_DATA}) train: {len(X_train)}, test: {len(X_test)}")
print(f"  Target ({TARGET_DATA}) train: {len(Y_train)}, test: {len(y_inds_test)}")
print(
    f"  Age bins: young=[{YOUNG_MIN_AGE}, {YOUNG_MAX_AGE}), "
    f"old=[{OLD_MIN_AGE}, {OLD_MAX_AGE})"
)

assert len(X_train) > 0, "No source samples found. Adjust young age range."
assert len(Y_train) > 0, "No target samples found. Adjust old age range."
assert len(X_test) > 0, "No source test samples found."

X_sampler = TensorSampler(X_train, device="cpu")
Y_sampler = TensorSampler(Y_train, device="cpu")

print("\n[2/4] Initialising LowRankDiag LightSB...")
torch.manual_seed(OUTPUT_SEED)
np.random.seed(OUTPUT_SEED)

D = LowRankDiagLightSBPotential(
    d=DIM,
    K=N_POTENTIALS,
    r=RANK,
    sigma=SIGMA,
    eps=EPS,
    init_delta=INIT_DELTA,
    init_mean_scale=INIT_MEAN_SCALE,
    alpha_scale=EPSILON,
).cpu()

with torch.no_grad():
    D.initialize_means_from_samples(Y_sampler.sample(N_POTENTIALS))
    if getattr(D, "r", 0) > 0:
        D.U.normal_(mean=0.0, std=INIT_U_STD)
    D.log_alpha_raw.zero_()
    D.recenter_alpha()

shape_params = [D.delta]
if getattr(D, "r", 0) > 0 and getattr(D.U, "requires_grad", False):
    shape_params.append(D.U)

D_opt = torch.optim.Adam(
    [
        {"params": [D.log_alpha_raw, D.m], "lr": D_LR_MAIN},
        {"params": shape_params, "lr": D_LR_SHAPE},
    ]
)

print(f"D optimizer lrs: {[g['lr'] for g in D_opt.param_groups]}")
assert all(g["lr"] > 0 for g in D_opt.param_groups), "Learning rate is zero or non-positive!"
print(f"Declared RANK: {RANK}, model.r: {getattr(D, 'r', None)}")
assert getattr(D, "r", None) == RANK, "Model rank does not match RANK setting"

with torch.no_grad():
    print(
        f"initial U_norm      = {D.U.norm().item():.6e}"
        if getattr(D, "r", 0) > 0 else "initial U_norm      = N/A"
    )
    if getattr(D, "r", 0) > 0:
        print(f"initial max|U|      = {D.U.abs().max().item():.6e}")
    print(f"initial S_diag_mean = {D.get_S().mean().item():.6e}")

print("\n[3/4] Training LowRankDiag LightSB...")
ckpt = os.path.join(OUTPUT_PATH, "D.pt")
if os.path.exists(ckpt):
    print("  Found checkpoint, loading and skipping training.")
    D.load_state_dict(torch.load(ckpt, map_location="cpu"))
else:
    wandb.init(
        name=EXP_NAME,
        config=dict(
            TASK=f"{INPUT_DATA}->{TARGET_DATA}",
            DIM=DIM,
            EPSILON=EPSILON,
            N_POTENTIALS=N_POTENTIALS,
            RANK=RANK,
            BATCH_SIZE=BATCH_SIZE,
            D_LR_MAIN=D_LR_MAIN,
            D_LR_SHAPE=D_LR_SHAPE,
            INIT_DELTA=INIT_DELTA,
            INIT_U_STD=INIT_U_STD,
            LAMBDA_DIAG=LAMBDA_DIAG,
            LAMBDA_U=LAMBDA_U,
            LAMBDA_M=LAMBDA_M,
            YOUNG_MIN_AGE=YOUNG_MIN_AGE,
            YOUNG_MAX_AGE=YOUNG_MAX_AGE,
            OLD_MIN_AGE=OLD_MIN_AGE,
            OLD_MAX_AGE=OLD_MAX_AGE,
        ),
        mode="disabled",
    )

    d_losses = []

    for step in tqdm(range(MAX_STEPS)):
        D_opt.zero_grad()

        X0 = X_sampler.sample(BATCH_SIZE)
        X1 = Y_sampler.sample(BATCH_SIZE)

        objective = D.training_loss(X0, X1)
        reg = D.regularization(
            lambda_diag=LAMBDA_DIAG,
            lambda_U=LAMBDA_U,
            lambda_m=LAMBDA_M,
        )
        D_loss = objective + reg
        D_loss.backward()

        u_grad_norm = 0.0
        if getattr(D, "r", 0) > 0:
            u_grad_norm = 0.0 if D.U.grad is None else D.U.grad.norm().item()

        grad_norm = torch.nn.utils.clip_grad_norm_(D.parameters(), D_GRADIENT_MAX_NORM)
        D_opt.step()

        wandb.log(
            {
                "D_loss": D_loss.item(),
                "objective": objective.item(),
                "reg": reg.item(),
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
                "U_norm": D.U.norm().item() if getattr(D, "r", 0) > 0 else 0.0,
                "U_grad_norm": u_grad_norm,
                "S_diag_mean": D.get_S().mean().item(),
            },
            step=step,
        )

        d_losses.append(float(D_loss.item()))

        if step % PRINT_EVERY == 0 or step == MAX_STEPS - 1:
            with torch.no_grad():
                if getattr(D, "r", 0) > 0:
                    print(
                        f"[step {step:5d}] "
                        f"loss={D_loss.item():+.6f} | "
                        f"obj={objective.item():+.6f} | "
                        f"reg={reg.item():.6e} | "
                        f"U_norm={D.U.norm().item():.6e} | "
                        f"U_grad_norm={u_grad_norm:.6e} | "
                        f"S_diag_mean={D.get_S().mean().item():.6e}"
                    )
                else:
                    print(
                        f"[step {step:5d}] "
                        f"loss={D_loss.item():+.6f} | "
                        f"obj={objective.item():+.6f} | "
                        f"reg={reg.item():.6e} | "
                        f"S_diag_mean={D.get_S().mean().item():.6e}"
                    )

    torch.save(D.state_dict(), os.path.join(OUTPUT_PATH, "D.pt"))
    torch.save(D_opt.state_dict(), os.path.join(OUTPUT_PATH, "D_opt.pt"))
    wandb.finish()
    print(f"  Saved to {OUTPUT_PATH}")

    try:
        loss_arr = np.array(d_losses)
        np.save(os.path.join(OUTPUT_PATH, "D_loss.npy"), loss_arr)
        plt.figure(figsize=(6, 3), dpi=150)
        plt.plot(loss_arr)
        plt.title("D training loss")
        plt.xlabel("step")
        plt.ylabel("D_loss")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, "D_loss.png"))
        plt.close()
        print(f"  Saved D loss curve to {OUTPUT_PATH}")
    except Exception as e:
        print(f"  Could not save D loss curve: {e}")

print("\n[4/4] Decoding results with ALAE...")
alae_cfg = os.path.join(ROOT, "ALAE", "configs", "ffhq.yaml")
alae_dir = os.path.join(ROOT, "ALAE", "training_artifacts", "ffhq")

os.chdir(os.path.join(ROOT, "ALAE"))
alae_model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
os.chdir(ROOT)

torch.manual_seed(OUTPUT_SEED)
np.random.seed(OUTPUT_SEED)

n_show = min(8, len(X_test))
n_samples = 3
max_img_idx = len(test_inp_imgs)
valid_pos = np.where(x_inds_test < max_img_idx)[0]
assert len(valid_pos) > 0, (
    "No valid test image indices after filtering. "
    "Check alignment between test_images.npy and test split indexing."
)

n_show = min(n_show, len(valid_pos))
inds_to_map = np.random.choice(valid_pos, size=n_show, replace=False)

lat_idx = x_inds_test[inds_to_map]
latent_to_map = torch.tensor(test_latents[lat_idx], dtype=torch.float32)
inp_images = test_inp_imgs[lat_idx]

with torch.no_grad():
    mapped = D.conditional_sample(latent_to_map, n_samples=n_samples)

try:
    alae_device = next(alae_model.parameters()).device
except StopIteration:
    alae_device = torch.device("cpu")

mapped = mapped.to(alae_device)

decoded_all = []
with torch.no_grad():
    for k in range(n_samples):
        img = decode(alae_model, mapped[:, k])
        img = ((img * 0.5 + 0.5) * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        decoded_all.append(img)
decoded_all = np.stack(decoded_all, axis=1)

fig, axes = plt.subplots(n_show, n_samples + 1, figsize=(n_samples + 1, n_show), dpi=200)
for i in range(n_show):
    axes[i][0].imshow(inp_images[i])
    axes[i][0].axis("off")
    for k in range(n_samples):
        axes[i][k + 1].imshow(decoded_all[i, k])
        axes[i][k + 1].axis("off")

axes[0][0].set_title("Input", fontsize=5)
for k in range(n_samples):
    axes[0][k + 1].set_title(f"Sample {k+1}", fontsize=5)

fig.suptitle(f"{INPUT_DATA} -> {TARGET_DATA}", fontsize=7, y=1.01)
fig.tight_layout(pad=0.05)

out_fig = os.path.join(OUTPUT_PATH, "result.png")
fig.savefig(out_fig, bbox_inches="tight")
print(f"  Result saved: {out_fig}")
plt.close()

print("\nDone!")
