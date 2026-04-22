"""
Run full LightSB + ALAE training (configurable).

This script is a thin, CLI-friendly wrapper around the logic in
`run_alae_experiment.py` but exposes `--max-steps`, `--batch-size`, and
`--device` so you can launch a full training run easily.
"""
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ALAE"))

from src.light_sb import LightSB
from lightsb_lr_diag import LowRankDiagLightSBPotential
from src.distributions import TensorSampler
from alae_ffhq_inference import load_model, encode, decode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--d-lr", type=float, default=1e-3)
    p.add_argument("--n-potentials", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--sampling-batch-size", type=int, default=128)
    p.add_argument("--rank", type=int, default=2, help="Low-rank dimension r")
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--init-delta", type=float, default=-2.0)
    p.add_argument("--init-mean-scale", type=float, default=0.2)
    p.add_argument("--lambda-diag", type=float, default=1e-4)
    p.add_argument("--lambda-U", type=float, default=1e-4)
    p.add_argument("--lambda-m", type=float, default=1e-5)
    # follow run_alae_experiment defaults
    p.add_argument(
        "--input-data",
        type=str,
        default="MAN",
        help=(
            "Source label "
            "(MAN/WOMAN/ADULT/CHILDREN/BOY/GIRL/YOUNG/OLD/"
            "YOUNG_MAN/OLD_MAN/YOUNG_WOMAN/OLD_WOMAN)"
        ),
    )
    p.add_argument(
        "--target-data",
        type=str,
        default="WOMAN",
        help=(
            "Target label "
            "(MAN/WOMAN/ADULT/CHILDREN/BOY/GIRL/YOUNG/OLD/"
            "YOUNG_MAN/OLD_MAN/YOUNG_WOMAN/OLD_WOMAN)"
        ),
    )
    p.add_argument("--young-min-age", type=int, default=18)
    p.add_argument("--young-max-age", type=int, default=30)
    p.add_argument("--old-min-age", type=int, default=55)
    p.add_argument("--old-max-age", type=int, default=100)
    p.add_argument("--init-by-samples", dest="init_by_samples", action="store_true")
    p.add_argument("--no-init-by-samples", dest="init_by_samples", action="store_false")
    p.set_defaults(init_by_samples=True)
    p.add_argument("--is-diagonal", dest="is_diagonal", action="store_true")
    p.add_argument("--no-is-diagonal", dest="is_diagonal", action="store_false")
    p.set_defaults(is_diagonal=True)
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if not set)")
    p.add_argument("--output-seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    return p.parse_args()


def main():
    args = parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_PATH = os.path.join(ROOT, "checkpoints", f"LightSB_ALAE_RUN")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Device selection
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.output_seed)
    np.random.seed(args.output_seed)

    # Load data
    data_dir = os.path.join(ROOT, "data")
    latents = np.load(os.path.join(data_dir, "latents.npy"))
    gender = np.load(os.path.join(data_dir, "gender.npy"))
    age = np.load(os.path.join(data_dir, "age.npy"))
    train_size, test_size = 60000, 10000
    train_latents, test_latents = latents[:train_size], latents[train_size:]
    train_gender, test_gender = gender[:train_size], gender[train_size:]
    train_age, test_age = age[:train_size], age[train_size:]

    def get_inds(g, a, label):
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
            return np.where(
                valid_age & (a >= args.young_min_age) & (a < args.young_max_age)
            )[0]
        elif label == "OLD":
            return np.where(
                valid_age & (a >= args.old_min_age) & (a < args.old_max_age)
            )[0]
        elif label == "YOUNG_MAN":
            return np.where(
                male & valid_age
                & (a >= args.young_min_age) & (a < args.young_max_age)
            )[0]
        elif label == "OLD_MAN":
            return np.where(
                male & valid_age
                & (a >= args.old_min_age) & (a < args.old_max_age)
            )[0]
        elif label == "YOUNG_WOMAN":
            return np.where(
                female & valid_age
                & (a >= args.young_min_age) & (a < args.young_max_age)
            )[0]
        elif label == "OLD_WOMAN":
            return np.where(
                female & valid_age
                & (a >= args.old_min_age) & (a < args.old_max_age)
            )[0]
        else:
            raise ValueError(f"Unknown label: {label}")

    x_inds_train = get_inds(train_gender, train_age, args.input_data)
    x_inds_test = get_inds(test_gender, test_age, args.input_data)
    y_inds_train = get_inds(train_gender, train_age, args.target_data)
    y_inds_test = get_inds(test_gender, test_age, args.target_data)

    X_train = torch.tensor(train_latents[x_inds_train])
    Y_train = torch.tensor(train_latents[y_inds_train])

    print(f"Source ({args.input_data}) train: {len(X_train)}, test: {len(x_inds_test)}")
    print(f"Target ({args.target_data}) train: {len(Y_train)}, test: {len(y_inds_test)}")
    print(
        f"Age bins: young=[{args.young_min_age}, {args.young_max_age}), "
        f"old=[{args.old_min_age}, {args.old_max_age})"
    )

    assert len(X_train) > 0, f"No source samples found for {args.input_data}."
    assert len(Y_train) > 0, f"No target samples found for {args.target_data}."
    assert len(x_inds_test) > 0, f"No source test samples found for {args.input_data}."

    X_sampler = TensorSampler(X_train, device=device)
    Y_sampler = TensorSampler(Y_train, device=device)

    # Model init
    # Use low-rank + diagonal LightSB potential
    model = LowRankDiagLightSBPotential(
        d=args.dim,
        K=args.n_potentials,
        r=args.rank,
        sigma=args.sigma,
        eps=args.eps,
        init_delta=args.init_delta,
        init_mean_scale=args.init_mean_scale,
        alpha_scale=args.epsilon,
    ).to(device)

    if args.init_by_samples:
        with torch.no_grad():
            model.initialize_means_from_samples(Y_sampler.sample(args.n_potentials))
            model.U.zero_()
            model.log_alpha_raw.zero_()
            model.recenter_alpha()

    # optimizer: mirror train_demo grouping
    model_opt = torch.optim.Adam(
        [
            {"params": [model.log_alpha_raw, model.m], "lr": 1e-3},
            {"params": [model.delta, model.U], "lr": 3e-4},
        ]
    )

    # Training (full)
    ckpt = os.path.join(OUTPUT_PATH, "D.pt")
    if os.path.exists(ckpt):
        print("Found checkpoint, loading and skipping training.")
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        if not args.wandb:
            import warnings
            warnings.filterwarnings("ignore")

        for step in tqdm(range(args.max_steps)):
            model_opt.zero_grad()
            X0 = X_sampler.sample(args.batch_size)
            X1 = Y_sampler.sample(args.batch_size)

            objective = model.training_loss(X0, X1)
            reg = model.regularization(
                lambda_diag=args.lambda_diag,
                lambda_U=args.lambda_U,
                lambda_m=args.lambda_m,
            )
            loss = objective + reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            model_opt.step()

        torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, "D.pt"))
        torch.save(model_opt.state_dict(), os.path.join(OUTPUT_PATH, "D_opt.pt"))
        print(f"Saved checkpoint to {OUTPUT_PATH}")

    # Decode with ALAE (same as original script)
    alae_cfg = os.path.join(ROOT, "ALAE", "configs", "ffhq.yaml")
    alae_dir = os.path.join(ROOT, "ALAE", "training_artifacts", "ffhq")
    os.chdir(os.path.join(ROOT, "ALAE"))
    alae_model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
    os.chdir(ROOT)

    # load test input images so we can save a visualization grid
    test_inp_path = os.path.join(ROOT, "data", "test_images.npy")
    test_inp_imgs = None
    if os.path.exists(test_inp_path):
        test_inp_imgs = np.load(test_inp_path)

    print("Decoding a few examples and saving result.png in checkpoint folder")
    # Minimal decode: pick a few test latents and map them
    torch.manual_seed(args.output_seed)
    # choose indices following run_alae_experiment.py: select from x_inds_test
    n_show = 8
    n_samples = 3
    test_pool = (x_inds_test < 300).sum()
    inds_to_map = np.random.choice(test_pool, size=n_show, replace=False)
    latent_to_map = torch.tensor(test_latents[x_inds_test[inds_to_map]], device=device)
    inp_images = test_inp_imgs[x_inds_test[inds_to_map]] if test_inp_imgs is not None else None

    # Sample conditional mappings from the LowRankDiag potential
    with torch.no_grad():
        # conditional_sample returns [B, n_samples, d]
        mapped = model.conditional_sample(latent_to_map, n_samples=n_samples)

    # Ensure mapped is on same device as ALAE model before decode
    try:
        alae_device = next(alae_model.parameters()).device
    except StopIteration:
        alae_device = torch.device('cpu')

    mapped = mapped.to(alae_device)

    # decode using ALAE helper
    decoded = []
    with torch.no_grad():
        for k in range(mapped.shape[1]):
            img = decode(alae_model, mapped[:, k])
            img = ((img * 0.5 + 0.5) * 255).clamp(0, 255).byte().permute(0,2,3,1).cpu().numpy()
            decoded.append(img)

    # stack decoded into shape (n_show, n_samples, H, W, 3)
    decoded_all = np.stack(decoded, axis=1)

    # Save figure similar to original script
    n_show = decoded_all.shape[0]
    n_samples = decoded_all.shape[1]
    fig, axes = plt.subplots(n_show, n_samples + 1,
                             figsize=(n_samples + 1, n_show), dpi=200)
    for i in range(n_show):
        if inp_images is not None:
            axes[i][0].imshow(inp_images[i])
        else:
            axes[i][0].axis('off')
        axes[i][0].axis("off")
        for k in range(n_samples):
            axes[i][k+1].imshow(decoded_all[i, k])
            axes[i][k+1].axis("off")

    axes[0][0].set_title("Input", fontsize=5)
    for k in range(n_samples):
        axes[0][k+1].set_title(f"Sample {k+1}", fontsize=5)

    fig.suptitle("Mapped → Decoded", fontsize=7, y=1.01)
    fig.tight_layout(pad=0.05)
    out_fig = os.path.join(OUTPUT_PATH, "result.png")
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Result saved: {out_fig}")

    print("Done.")


if __name__ == "__main__":
    main()
