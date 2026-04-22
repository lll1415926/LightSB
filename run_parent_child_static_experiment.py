"""
Static parent-child LightSB-style experiment.

This script trains a conditional static Schrödinger / entropic OT model
from joint parent latents (father, mother) to child latents.

Supported inputs:
1. A user-provided `.npz` file with arrays:
       z_f: [N, d]
       z_m: [N, d]
       y:   [N, d]
   and optionally:
       a:   [N] integer condition labels
       split: [N] with values in {"train", "test"} or {0, 1}
2. Synthetic data via `--synthetic`, useful for smoke tests.
"""
import argparse
import os
import sys
from typing import Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.conditional_static_lightsb import StaticParentChildLightSB


ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default="")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic-size", type=int, default=12000)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--n-conditions", type=int, default=2)
    p.add_argument("--use-interaction", action="store_true")
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--n-potentials", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--print-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument(
        "--select-best-by",
        type=str,
        default="mse",
        choices=["mse", "loss"],
    )
    p.add_argument("--output-seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--exp-suffix", type=str, default="")
    p.add_argument("--force-retrain", action="store_true")
    return p.parse_args()


def _split_mask(raw_split: np.ndarray) -> np.ndarray:
    if raw_split.dtype.kind in ("U", "S", "O"):
        values = np.asarray(raw_split).astype(str)
        return np.char.lower(values) == "train"
    return np.asarray(raw_split) == 0


def load_npz_dataset(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = ["z_f", "z_m", "y"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required array '{key}' in {path}")

    z_f = np.asarray(data["z_f"], dtype=np.float32)
    z_m = np.asarray(data["z_m"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    if z_f.shape != z_m.shape:
        raise ValueError("z_f and z_m must have the same shape.")
    if z_f.shape != y.shape:
        raise ValueError("z_f, z_m, and y must share the same shape [N, d].")

    if "a" in data:
        a = np.asarray(data["a"]).reshape(-1).astype(np.int64)
    else:
        a = np.zeros(z_f.shape[0], dtype=np.int64)

    if "split" in data:
        train_mask = _split_mask(np.asarray(data["split"]).reshape(-1))
    else:
        n = z_f.shape[0]
        n_train = max(1, int(0.9 * n))
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:n_train] = True

    test_mask = ~train_mask
    if not test_mask.any():
        test_mask[-max(1, z_f.shape[0] // 10):] = True
        train_mask = ~test_mask

    return {
        "z_f_train": z_f[train_mask],
        "z_m_train": z_m[train_mask],
        "y_train": y[train_mask],
        "a_train": a[train_mask],
        "z_f_test": z_f[test_mask],
        "z_m_test": z_m[test_mask],
        "y_test": y[test_mask],
        "a_test": a[test_mask],
    }


def build_synthetic_dataset(
    n: int,
    latent_dim: int,
    n_conditions: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z_f = rng.normal(size=(n, latent_dim)).astype(np.float32)
    z_m = rng.normal(size=(n, latent_dim)).astype(np.float32)
    a = rng.integers(0, n_conditions, size=n, dtype=np.int64)

    W_f = rng.normal(scale=0.5, size=(latent_dim, latent_dim)).astype(np.float32)
    W_m = rng.normal(scale=0.5, size=(latent_dim, latent_dim)).astype(np.float32)
    W_fm = rng.normal(scale=0.2, size=(latent_dim, latent_dim)).astype(np.float32)
    cond_bias = rng.normal(scale=0.35, size=(n_conditions, latent_dim)).astype(np.float32)

    y_mean = (
        z_f @ W_f.T
        + z_m @ W_m.T
        + (z_f * z_m) @ W_fm.T
        + cond_bias[a]
    )
    y = (y_mean + 0.3 * rng.normal(size=y_mean.shape)).astype(np.float32)

    n_train = max(1, int(0.9 * n))
    return {
        "z_f_train": z_f[:n_train],
        "z_m_train": z_m[:n_train],
        "y_train": y[:n_train],
        "a_train": a[:n_train],
        "z_f_test": z_f[n_train:],
        "z_m_test": z_m[n_train:],
        "y_test": y[n_train:],
        "a_test": a[n_train:],
    }


def sample_batch(
    arrays: Tuple[torch.Tensor, ...],
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    n = arrays[0].shape[0]
    idx = torch.randint(0, n, (batch_size,), device=device)
    return tuple(arr[idx] for arr in arrays)


@torch.no_grad()
def evaluate_model(
    model: StaticParentChildLightSB,
    z_f: torch.Tensor,
    z_m: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    batch_size: int = 1024,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_count = 0
    for start in range(0, z_f.shape[0], batch_size):
        end = min(start + batch_size, z_f.shape[0])
        zf_b = z_f[start:end]
        zm_b = z_m[start:end]
        y_b = y[start:end]
        a_b = a[start:end]
        loss = model.loss(zf_b, zm_b, y_b, a_b)
        pred_mean = model.sample_child(zf_b, zm_b, a_b, n_samples=1)
        mse = (pred_mean - y_b).square().mean()

        bsz = end - start
        total_loss += float(loss.item()) * bsz
        total_mse += float(mse.item()) * bsz
        total_count += bsz

    return {
        "loss": total_loss / max(total_count, 1),
        "mse": total_mse / max(total_count, 1),
    }


def save_preview(
    model: StaticParentChildLightSB,
    test_data: Dict[str, torch.Tensor],
    output_dir: str,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_show = min(8, test_data["z_f"].shape[0])
    if n_show == 0:
        return

    inds = np.random.choice(test_data["z_f"].shape[0], size=n_show, replace=False)
    zf = test_data["z_f"][inds]
    zm = test_data["z_m"][inds]
    y_true = test_data["y"][inds]
    a = test_data["a"][inds]

    samples = model.sample_child(zf, zm, a, n_samples=3).cpu().numpy()
    y_true = y_true.cpu().numpy()
    zf = zf.cpu().numpy()
    zm = zm.cpu().numpy()
    a = a.cpu().numpy()

    np.savez(
        os.path.join(output_dir, "preview_samples.npz"),
        z_f=zf,
        z_m=zm,
        y_true=y_true,
        y_samples=samples,
        a=a,
    )

    preview_dim = min(4, y_true.shape[1])
    fig, axes = plt.subplots(preview_dim, 1, figsize=(9, 2.2 * preview_dim), dpi=150)
    if preview_dim == 1:
        axes = [axes]
    for dim_idx, ax in enumerate(axes):
        ax.plot(y_true[:, dim_idx], label="true", marker="o")
        ax.plot(samples[:, 0, dim_idx], label="sample_1", marker="x")
        ax.plot(samples[:, 1, dim_idx], label="sample_2", marker="^")
        ax.set_title(f"Preview latent dim {dim_idx}")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "preview_plot.png"))
    plt.close(fig)


def main():
    args = parse_args()

    exp_name = f"ParentChildStaticLightSB_DIM_{args.latent_dim}_EPS_{args.epsilon}"
    if args.use_interaction:
        exp_name += "_interaction"
    if args.exp_suffix.strip():
        exp_name += f"_{args.exp_suffix.strip()}"
    output_dir = os.path.join(ROOT, "checkpoints", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== {exp_name} ===")
    print(f"Output: {output_dir}")

    torch.manual_seed(args.output_seed)
    np.random.seed(args.output_seed)

    if args.synthetic:
        dataset = build_synthetic_dataset(
            n=args.synthetic_size,
            latent_dim=args.latent_dim,
            n_conditions=args.n_conditions,
            seed=args.output_seed,
        )
        print("Using synthetic parent-child latent dataset.")
    else:
        if not args.data_path:
            raise ValueError("Provide --data-path or use --synthetic.")
        dataset = load_npz_dataset(args.data_path)
        print(f"Loaded dataset: {args.data_path}")

    train_zf = torch.tensor(dataset["z_f_train"], dtype=torch.float32)
    train_zm = torch.tensor(dataset["z_m_train"], dtype=torch.float32)
    train_y = torch.tensor(dataset["y_train"], dtype=torch.float32)
    train_a = torch.tensor(dataset["a_train"], dtype=torch.long)
    test_zf = torch.tensor(dataset["z_f_test"], dtype=torch.float32)
    test_zm = torch.tensor(dataset["z_m_test"], dtype=torch.float32)
    test_y = torch.tensor(dataset["y_test"], dtype=torch.float32)
    test_a = torch.tensor(dataset["a_test"], dtype=torch.long)

    if train_zf.shape[1] != args.latent_dim:
        raise ValueError(
            f"Latent dim mismatch: args.latent_dim={args.latent_dim}, "
            f"but dataset has {train_zf.shape[1]}"
        )

    inferred_conditions = int(max(train_a.max().item(), test_a.max().item()) + 1)
    num_conditions = max(args.n_conditions, inferred_conditions)

    print(
        f"Train size={len(train_zf)}, Test size={len(test_zf)}, "
        f"Latent dim={args.latent_dim}, Conditions={num_conditions}"
    )

    device = torch.device("cpu")
    model = StaticParentChildLightSB(
        latent_dim=args.latent_dim,
        n_components=args.n_potentials,
        epsilon=args.epsilon,
        num_conditions=num_conditions,
        use_interaction=args.use_interaction,
    ).to(device)

    ckpt_path = os.path.join(output_dir, "model.pt")
    if os.path.exists(ckpt_path) and not args.force_retrain:
        print("Found checkpoint, skipping training.")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        loss_history = []
        eval_history = []
        best_eval = float("inf")
        best_state = None

        train_tensors = tuple(
            arr.to(device) for arr in (train_zf, train_zm, train_y, train_a)
        )
        test_tensors = {
            "z_f": test_zf.to(device),
            "z_m": test_zm.to(device),
            "y": test_y.to(device),
            "a": test_a.to(device),
        }

        print("Training static conditional LightSB model...")
        for step in tqdm(range(args.max_steps)):
            model.train()
            zf_b, zm_b, y_b, a_b = sample_batch(train_tensors, args.batch_size, device)

            optimizer.zero_grad()
            loss = model.loss(zf_b, zm_b, y_b, a_b)
            loss.backward()
            optimizer.step()

            loss_history.append(float(loss.item()))

            if (step + 1) % args.eval_every == 0 or step == 0:
                metrics = evaluate_model(
                    model=model,
                    z_f=test_tensors["z_f"],
                    z_m=test_tensors["z_m"],
                    y=test_tensors["y"],
                    a=test_tensors["a"],
                )
                eval_history.append([step + 1, metrics["loss"], metrics["mse"]])
                selection_metric = metrics[args.select_best_by]
                if selection_metric < best_eval:
                    best_eval = selection_metric
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                if (step + 1) % args.print_every == 0 or step == 0:
                    print(
                        f"step={step + 1:5d} "
                        f"train_loss={loss.item():.4f} "
                        f"test_loss={metrics['loss']:.4f} "
                        f"test_mse={metrics['mse']:.4f}"
                    )

        if best_state is not None:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), ckpt_path)
        np.save(os.path.join(output_dir, "train_loss.npy"), np.array(loss_history))
        np.save(os.path.join(output_dir, "eval_metrics.npy"), np.array(eval_history))

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=150)
        axes[0].plot(np.array(loss_history))
        axes[0].set_title("Train loss")
        axes[0].set_xlabel("step")
        axes[0].grid(alpha=0.25)
        eval_arr = np.array(eval_history)
        if len(eval_arr) > 0:
            axes[1].plot(eval_arr[:, 0], eval_arr[:, 1], label="test_loss")
            axes[1].plot(eval_arr[:, 0], eval_arr[:, 2], label="test_mse")
            axes[1].legend()
        axes[1].set_title("Eval metrics")
        axes[1].set_xlabel("step")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "training_curves.png"))
        plt.close(fig)

    final_metrics = evaluate_model(
        model=model,
        z_f=test_zf,
        z_m=test_zm,
        y=test_y,
        a=test_a,
    )
    print(
        f"Final test metrics: loss={final_metrics['loss']:.4f}, "
        f"mse={final_metrics['mse']:.4f}"
    )

    save_preview(
        model=model,
        test_data={
            "z_f": test_zf,
            "z_m": test_zm,
            "y": test_y,
            "a": test_a,
        },
        output_dir=output_dir,
        seed=args.output_seed,
    )
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
