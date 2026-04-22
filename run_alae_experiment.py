"""
LightSB ALAE Image Translation Experiment
Supports gender- and age-conditioned FFHQ translations in ALAE latent space.
"""
import os, sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALAE"))

import torch
import numpy as np
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt

from src.light_sb import LightSB
from src.distributions import TensorSampler
from alae_ffhq_inference import load_model, encode, decode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-data", type=str, default="YOUNG_MAN")
    p.add_argument("--target-data", type=str, default="OLD_MAN")
    p.add_argument("--young-min-age", type=int, default=18)
    p.add_argument("--young-max-age", type=int, default=30)
    p.add_argument("--old-min-age", type=int, default=55)
    p.add_argument("--old-max-age", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-potentials", type=int, default=10)
    p.add_argument("--d-lr", type=float, default=1e-3)
    p.add_argument("--output-seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--exp-suffix", type=str, default="")
    p.add_argument("--force-retrain", action="store_true")
    return p.parse_args()


args = parse_args()

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT           = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA     = args.input_data
TARGET_DATA    = args.target_data
YOUNG_MIN_AGE  = args.young_min_age
YOUNG_MAX_AGE  = args.young_max_age   # [18, 30)
OLD_MIN_AGE    = args.old_min_age
OLD_MAX_AGE    = args.old_max_age  # [55, 100)
DIM            = 512
EPSILON        = args.epsilon
N_POTENTIALS   = args.n_potentials
BATCH_SIZE     = args.batch_size
D_LR           = args.d_lr
D_GRADIENT_MAX_NORM = float("inf")
SAMPLING_BATCH_SIZE = 128
INIT_BY_SAMPLES = True
IS_DIAGONAL    = True
MAX_STEPS      = args.max_steps
OUTPUT_SEED    = args.output_seed

EXP_NAME   = f"LightSB_ALAE_{INPUT_DATA}_TO_{TARGET_DATA}_EPSILON_{EPSILON}"
if args.exp_suffix:
    EXP_NAME = f"{EXP_NAME}_{args.exp_suffix}"
OUTPUT_PATH = os.path.join(ROOT, "checkpoints", EXP_NAME)
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"=== {EXP_NAME} ===")
print(f"Output: {OUTPUT_PATH}")

# ─── Data ─────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading data...")
data_dir = os.path.join(ROOT, "data")
train_size, test_size = 60000, 10000

latents       = np.load(os.path.join(data_dir, "latents.npy"))
gender        = np.load(os.path.join(data_dir, "gender.npy"))
age           = np.load(os.path.join(data_dir, "age.npy"))
test_inp_imgs = np.load(os.path.join(data_dir, "test_images.npy"))

train_latents, test_latents = latents[:train_size], latents[train_size:]
train_gender,  test_gender  = gender[:train_size],  gender[train_size:]
train_age,     test_age     = age[:train_size],     age[train_size:]

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
x_inds_test  = get_inds(test_gender,  test_age,  "test",  INPUT_DATA)
y_inds_train = get_inds(train_gender, train_age, "train", TARGET_DATA)
y_inds_test  = get_inds(test_gender,  test_age,  "test",  TARGET_DATA)

X_train = torch.tensor(train_latents[x_inds_train])
Y_train = torch.tensor(train_latents[y_inds_train])
X_test  = torch.tensor(test_latents[x_inds_test])

print(f"  Source ({INPUT_DATA}) train: {len(X_train)}, test: {len(X_test)}")
print(f"  Target ({TARGET_DATA}) train: {len(Y_train)}")
print(
    f"  Age bins: young=[{YOUNG_MIN_AGE}, {YOUNG_MAX_AGE}), "
    f"old=[{OLD_MIN_AGE}, {OLD_MAX_AGE})"
)

assert len(X_train) > 0, f"No source samples found for {INPUT_DATA}."
assert len(Y_train) > 0, f"No target samples found for {TARGET_DATA}."
assert len(X_test) > 0, f"No source test samples found for {INPUT_DATA}."

X_sampler = TensorSampler(X_train, device="cpu")
Y_sampler = TensorSampler(Y_train, device="cpu")

# ─── Model init ───────────────────────────────────────────────────────────────
print("\n[2/4] Initialising LightSB...")
torch.manual_seed(OUTPUT_SEED); np.random.seed(OUTPUT_SEED)

D = LightSB(dim=DIM, n_potentials=N_POTENTIALS, epsilon=EPSILON,
            sampling_batch_size=SAMPLING_BATCH_SIZE,
            S_diagonal_init=0.1, is_diagonal=IS_DIAGONAL).cpu()

if INIT_BY_SAMPLES:
    D.init_r_by_samples(Y_sampler.sample(N_POTENTIALS))

D_opt = torch.optim.Adam(D.parameters(), lr=D_LR)

# ─── Train ────────────────────────────────────────────────────────────────────
print("\n[3/4] Training LightSB...")
ckpt = os.path.join(OUTPUT_PATH, "D.pt")
if os.path.exists(ckpt) and not args.force_retrain:
    print(f"  Found checkpoint, skipping training.")
    D.load_state_dict(torch.load(ckpt, map_location="cpu"))
else:
    wandb.init(name=EXP_NAME, config=dict(
        DIM=DIM, D_LR=D_LR, BATCH_SIZE=BATCH_SIZE, EPSILON=EPSILON,
        N_POTENTIALS=N_POTENTIALS, IS_DIAGONAL=IS_DIAGONAL,
    ), mode="disabled")

    # record D loss per step for later plotting
    d_losses = []

    for step in tqdm(range(MAX_STEPS)):
        D_opt.zero_grad()
        X0 = X_sampler.sample(BATCH_SIZE)
        X1 = Y_sampler.sample(BATCH_SIZE)

        log_potential = D.get_log_potential(X1)
        log_C         = D.get_log_C(X0)
        D_loss        = (-log_potential + log_C).mean()

        D_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(D.parameters(), D_GRADIENT_MAX_NORM)
        D_opt.step()

        wandb.log({"D_loss": D_loss.item(), "grad_norm": grad_norm.item()}, step=step)
        try:
            d_losses.append(float(D_loss.item()))
        except Exception:
            d_losses.append(D_loss.item())

    torch.save(D.state_dict(),     os.path.join(OUTPUT_PATH, "D.pt"))
    torch.save(D_opt.state_dict(), os.path.join(OUTPUT_PATH, "D_opt.pt"))
    wandb.finish()
    print(f"  Saved to {OUTPUT_PATH}")
    # save D loss curve and numpy array
    try:
        import matplotlib.pyplot as _plt
        loss_arr = np.array(d_losses)
        np.save(os.path.join(OUTPUT_PATH, "D_loss.npy"), loss_arr)
        _plt.figure(figsize=(6, 3), dpi=150)
        _plt.plot(loss_arr)
        _plt.title('D training loss')
        _plt.xlabel('step')
        _plt.ylabel('D_loss')
        _plt.tight_layout()
        _plt.savefig(os.path.join(OUTPUT_PATH, "D_loss.png"))
        _plt.close()
        print(f"  Saved D loss curve to {OUTPUT_PATH}")
    except Exception as e:
        print(f"  Could not save D loss curve: {e}")

# ─── Results ──────────────────────────────────────────────────────────────────
print("\n[4/4] Decoding results with ALAE...")
alae_cfg = os.path.join(ROOT, "ALAE", "configs", "ffhq.yaml")
alae_dir = os.path.join(ROOT, "ALAE", "training_artifacts", "ffhq")
# checkpointer uses relative paths, must run from ALAE dir
os.chdir(os.path.join(ROOT, "ALAE"))
model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
os.chdir(ROOT)

torch.manual_seed(OUTPUT_SEED); np.random.seed(OUTPUT_SEED)

n_show        = 8
n_samples     = 3
test_pool     = (x_inds_test < 300).sum()
inds_to_map   = np.random.choice(test_pool, size=n_show, replace=False)
latent_to_map = torch.tensor(test_latents[x_inds_test[inds_to_map]])
inp_images    = test_inp_imgs[x_inds_test[inds_to_map]]

with torch.no_grad():
    mapped_all = [D(latent_to_map.cpu()) for _ in range(n_samples)]
mapped = torch.stack(mapped_all, dim=1)   # (n_show, n_samples, dim)

decoded_all = []
with torch.no_grad():
    for k in range(n_samples):
        img = decode(model, mapped[:, k])
        img = ((img * 0.5 + 0.5) * 255).clamp(0, 255).byte().permute(0,2,3,1).cpu().numpy()
        decoded_all.append(img)
decoded_all = np.stack(decoded_all, axis=1)   # (n_show, n_samples, H, W, 3)

# Save figure
fig, axes = plt.subplots(n_show, n_samples + 1,
                         figsize=(n_samples + 1, n_show), dpi=200)
for i in range(n_show):
    axes[i][0].imshow(inp_images[i])
    axes[i][0].axis("off")
    for k in range(n_samples):
        axes[i][k+1].imshow(decoded_all[i, k])
        axes[i][k+1].axis("off")

axes[0][0].set_title("Input", fontsize=5)
for k in range(n_samples):
    axes[0][k+1].set_title(f"Sample {k+1}", fontsize=5)

fig.suptitle(f"{INPUT_DATA} → {TARGET_DATA}", fontsize=7, y=1.01)
fig.tight_layout(pad=0.05)

out_fig = os.path.join(OUTPUT_PATH, "result.png")
fig.savefig(out_fig, bbox_inches="tight")
print(f"  Result saved: {out_fig}")
plt.close()
print("\nDone!")
