"""
Static 2d -> d EOT parent-child experiment in ALAE latent space.

Key idea:
    - Sample father, mother, and child latents independently from
      adult-male, adult-female, and children subsets of the ALAE FFHQ latents.
    - Build the source variable as:
          x = [z_f, z_m] in R^{2d}
    - Train the static EOT objective:
          E_x[log Z_theta(x)] - E_y[log eta_theta(y)]
      with the closed-form Gaussian-mixture formulas induced by the cost
          c_{B,M}(x,y) = 1/2 (Bx - y)^T M (Bx - y).

This is *not* supervised family-triplet training. It is a marginal/static
entropic-OT style experiment over:

    (adult male latent, adult female latent)  -->  child latent.
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
from src.conditional_static_lightsb import StaticParentChildLightSB
from src.distributions import TensorSampler


ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SIZE = 60000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adult-min-age", type=int, default=25)
    p.add_argument("--adult-max-age", type=int, default=55)
    p.add_argument("--child-max-age", type=int, default=12)
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--n-potentials", type=int, default=10)
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument(
        "--b-mode",
        type=str,
        default="diag_gate",
        choices=["average", "first", "random", "diag_gate"],
    )
    p.add_argument("--freeze-b", action="store_true")
    p.add_argument("--b-dev-lambda", type=float, default=1e-3)
    p.add_argument("--learn-m", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--print-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-train-samples", action="store_true")
    p.add_argument("--select-best-by", type=str, default="loss", choices=["loss", "mse"])
    p.add_argument("--n-show", type=int, default=8)
    p.add_argument("--n-final-samples", type=int, default=3)
    p.add_argument("--output-seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--sample-seed", type=lambda x: int(x, 0), default=None)
    p.add_argument("--exp-suffix", type=str, default="")
    p.add_argument("--force-retrain", action="store_true")
    return p.parse_args()


def get_inds(g, a, label, adult_min_age, adult_max_age, child_max_age):
    g = g.reshape(-1)
    a = a.reshape(-1)

    valid_age = a != -1
    male = g == "male"
    female = g == "female"

    if label == "ADULT_MAN":
        return np.where(
            male & valid_age & (a >= adult_min_age) & (a < adult_max_age)
        )[0]
    if label == "ADULT_WOMAN":
        return np.where(
            female & valid_age & (a >= adult_min_age) & (a < adult_max_age)
        )[0]
    if label == "CHILDREN":
        return np.where(valid_age & (a < child_max_age))[0]
    raise ValueError(f"Unknown label: {label}")


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


@torch.no_grad()
def render_preview_grid(
    model,
    alae_model,
    father_pool,
    mother_pool,
    father_indices_global,
    mother_indices_global,
    output_dir,
    prefix,
    seed,
    n_show,
    n_samples,
):
    rng = np.random.default_rng(seed)
    father_show_idx = rng.choice(len(father_pool), size=n_show, replace=False)
    mother_show_idx = rng.choice(len(mother_pool), size=n_show, replace=False)

    father_latents = father_pool[father_show_idx]
    mother_latents = mother_pool[mother_show_idx]

    father_imgs = decode_batch(alae_model, father_latents)
    mother_imgs = decode_batch(alae_model, mother_latents)
    mapped = model.sample_child(
        father_latents,
        mother_latents,
        None,
        n_samples=n_samples,
    )

    decoded_all = []
    for k in range(n_samples):
        img = decode_batch(alae_model, mapped[:, k])
        decoded_all.append(img)
    decoded_all = np.stack(decoded_all, axis=1)

    np.savez(
        os.path.join(output_dir, f"{prefix}_samples.npz"),
        father_latents=father_latents.numpy(),
        mother_latents=mother_latents.numpy(),
        sampled_children=mapped.numpy(),
        father_indices=father_indices_global[father_show_idx],
        mother_indices=mother_indices_global[mother_show_idx],
    )

    fig, axes = plt.subplots(n_show, n_samples + 2, figsize=(n_samples + 2, n_show), dpi=200)
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(n_show):
        axes[i][0].imshow(father_imgs[i])
        axes[i][0].axis("off")
        axes[i][1].imshow(mother_imgs[i])
        axes[i][1].axis("off")
        for k in range(n_samples):
            axes[i][k + 2].imshow(decoded_all[i, k])
            axes[i][k + 2].axis("off")

    axes[0][0].set_title("Father", fontsize=5)
    axes[0][1].set_title("Mother", fontsize=5)
    for k in range(n_samples):
        axes[0][k + 2].set_title(f"Sample {k + 1}", fontsize=5)

    fig.tight_layout(pad=0.05)
    fig.savefig(os.path.join(output_dir, f"{prefix}.png"), bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_model(model, father, mother, child, batch_size=1024):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_count = 0
    for start in range(0, father.shape[0], batch_size):
        end = min(start + batch_size, father.shape[0])
        f_b = father[start:end]
        m_b = mother[start:end]
        c_b = child[start:end]
        loss = model.loss(f_b, m_b, c_b, None)
        pred = model.sample_child(f_b, m_b, None, n_samples=1)
        mse = (pred - c_b).square().mean()

        bsz = end - start
        total_loss += float(loss.item()) * bsz
        total_mse += float(mse.item()) * bsz
        total_count += bsz

    return {
        "loss": total_loss / max(total_count, 1),
        "mse": total_mse / max(total_count, 1),
    }


def main():
    args = parse_args()
    sample_seed = args.output_seed if args.sample_seed is None else args.sample_seed

    exp_name = (
        f"LightSB_ALAE_PARENT_CHILD_MARGINAL"
        f"_EPS_{args.epsilon}"
        f"_K_{args.n_potentials}"
        f"_B_{args.b_mode}"
    )
    if args.freeze_b:
        exp_name += "_freezeB"
    elif args.b_dev_lambda > 0.0:
        exp_name += f"_Bdev_{args.b_dev_lambda:g}"
    if args.learn_m:
        exp_name += "_learnM"
    if args.exp_suffix.strip():
        exp_name += f"_{args.exp_suffix.strip()}"
    output_dir = os.path.join(ROOT, "checkpoints", exp_name)
    os.makedirs(output_dir, exist_ok=True)
    sample_dir = os.path.join(output_dir, "training_samples")
    if args.save_train_samples:
        os.makedirs(sample_dir, exist_ok=True)

    print(f"=== {exp_name} ===")
    print(f"Output: {output_dir}")

    torch.manual_seed(args.output_seed)
    np.random.seed(args.output_seed)

    data_dir = os.path.join(ROOT, "data")
    latents = np.load(os.path.join(data_dir, "latents.npy"))
    gender = np.load(os.path.join(data_dir, "gender.npy"))
    age = np.load(os.path.join(data_dir, "age.npy"))
    train_latents, test_latents = latents[:TRAIN_SIZE], latents[TRAIN_SIZE:]
    train_gender, test_gender = gender[:TRAIN_SIZE], gender[TRAIN_SIZE:]
    train_age, test_age = age[:TRAIN_SIZE], age[TRAIN_SIZE:]

    father_train_idx = get_inds(
        train_gender, train_age, "ADULT_MAN",
        args.adult_min_age, args.adult_max_age, args.child_max_age
    )
    mother_train_idx = get_inds(
        train_gender, train_age, "ADULT_WOMAN",
        args.adult_min_age, args.adult_max_age, args.child_max_age
    )
    child_train_idx = get_inds(
        train_gender, train_age, "CHILDREN",
        args.adult_min_age, args.adult_max_age, args.child_max_age
    )

    father_test_idx = get_inds(
        test_gender, test_age, "ADULT_MAN",
        args.adult_min_age, args.adult_max_age, args.child_max_age
    )
    mother_test_idx = get_inds(
        test_gender, test_age, "ADULT_WOMAN",
        args.adult_min_age, args.adult_max_age, args.child_max_age
    )
    child_test_idx = get_inds(
        test_gender, test_age, "CHILDREN",
        args.adult_min_age, args.adult_max_age, args.child_max_age
    )

    father_train = torch.tensor(train_latents[father_train_idx], dtype=torch.float32)
    mother_train = torch.tensor(train_latents[mother_train_idx], dtype=torch.float32)
    child_train = torch.tensor(train_latents[child_train_idx], dtype=torch.float32)

    father_test = torch.tensor(test_latents[father_test_idx], dtype=torch.float32)
    mother_test = torch.tensor(test_latents[mother_test_idx], dtype=torch.float32)
    child_test = torch.tensor(test_latents[child_test_idx], dtype=torch.float32)

    print(
        f"Train sets: father={len(father_train)}, mother={len(mother_train)}, child={len(child_train)}"
    )
    print(
        f"Test sets: father={len(father_test)}, mother={len(mother_test)}, child={len(child_test)}"
    )
    print(
        f"Age bins: adults=[{args.adult_min_age}, {args.adult_max_age}), children=[0, {args.child_max_age})"
    )

    assert len(father_train) > 0
    assert len(mother_train) > 0
    assert len(child_train) > 0
    assert len(father_test) > 0
    assert len(mother_test) > 0
    assert len(child_test) > 0

    father_sampler = TensorSampler(father_train, device="cpu")
    mother_sampler = TensorSampler(mother_train, device="cpu")
    child_sampler = TensorSampler(child_train, device="cpu")

    alae_cfg = os.path.join(ROOT, "ALAE", "configs", "ffhq.yaml")
    alae_dir = os.path.join(ROOT, "ALAE", "training_artifacts", "ffhq")
    os.chdir(os.path.join(ROOT, "ALAE"))
    alae_model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
    os.chdir(ROOT)

    model = StaticParentChildLightSB(
        latent_dim=args.latent_dim,
        n_components=args.n_potentials,
        epsilon=args.epsilon,
        init_scale=args.init_scale,
        b_mode=args.b_mode,
        freeze_b=args.freeze_b,
        learn_m=args.learn_m,
    ).cpu()

    with torch.no_grad():
        init_samples = child_sampler.sample(args.n_potentials)
        model.init_r_by_samples(init_samples)

    ckpt_path = os.path.join(output_dir, "model.pt")
    if os.path.exists(ckpt_path) and not args.force_retrain:
        print("Found checkpoint, loading and skipping training.")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
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

        test_eval_size = min(len(father_test), len(mother_test), len(child_test))
        rng_eval = np.random.default_rng(args.output_seed)
        father_eval_idx = rng_eval.choice(len(father_test), size=test_eval_size, replace=False)
        mother_eval_idx = rng_eval.choice(len(mother_test), size=test_eval_size, replace=False)
        child_eval_idx = rng_eval.choice(len(child_test), size=test_eval_size, replace=False)

        father_eval = father_test[father_eval_idx]
        mother_eval = mother_test[mother_eval_idx]
        child_eval = child_test[child_eval_idx]

        print("Training parent-child marginal LightSB...")
        for step in tqdm(range(args.max_steps)):
            model.train()
            f_b = father_sampler.sample(args.batch_size)
            m_b = mother_sampler.sample(args.batch_size)
            c_b = child_sampler.sample(args.batch_size)

            optimizer.zero_grad()
            data_loss = model.loss(f_b, m_b, c_b, None)
            b_reg = args.b_dev_lambda * model.b_deviation_penalty()
            loss = data_loss + b_reg
            loss.backward()
            optimizer.step()

            loss_history.append(float(loss.item()))

            if (step + 1) % args.eval_every == 0 or step == 0:
                metrics = evaluate_model(model, father_eval, mother_eval, child_eval)
                eval_history.append([step + 1, metrics["loss"], metrics["mse"]])
                selection_metric = metrics[args.select_best_by]
                if selection_metric < best_eval:
                    best_eval = selection_metric
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                if args.save_train_samples:
                    render_preview_grid(
                        model=model,
                        alae_model=alae_model,
                        father_pool=father_test,
                        mother_pool=mother_test,
                        father_indices_global=father_test_idx,
                        mother_indices_global=mother_test_idx,
                        output_dir=sample_dir,
                        prefix=f"step_{step + 1:05d}",
                        seed=args.output_seed + step + 1,
                        n_show=args.n_show,
                        n_samples=args.n_final_samples,
                    )
                if (step + 1) % args.print_every == 0 or step == 0:
                    print(
                        f"step={step + 1:5d} "
                        f"train_loss={loss.item():.4f} "
                        f"data_loss={data_loss.item():.4f} "
                        f"b_reg={b_reg.item():.4f} "
                        f"eval_loss={metrics['loss']:.4f} "
                        f"eval_mse={metrics['mse']:.4f}"
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
            axes[1].plot(eval_arr[:, 0], eval_arr[:, 1], label="eval_loss")
            axes[1].plot(eval_arr[:, 0], eval_arr[:, 2], label="eval_mse")
            axes[1].legend()
        axes[1].set_title("Eval metrics")
        axes[1].set_xlabel("step")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "training_curves.png"))
        plt.close(fig)

    final_eval_size = min(len(father_test), len(mother_test), len(child_test))
    rng_final = np.random.default_rng(args.output_seed + 1)
    father_eval_idx = rng_final.choice(len(father_test), size=final_eval_size, replace=False)
    mother_eval_idx = rng_final.choice(len(mother_test), size=final_eval_size, replace=False)
    child_eval_idx = rng_final.choice(len(child_test), size=final_eval_size, replace=False)

    final_metrics = evaluate_model(
        model,
        father_test[father_eval_idx],
        mother_test[mother_eval_idx],
        child_test[child_eval_idx],
    )
    print(
        f"Final eval metrics: loss={final_metrics['loss']:.4f}, "
        f"mse={final_metrics['mse']:.4f}"
    )
    render_preview_grid(
        model=model,
        alae_model=alae_model,
        father_pool=father_test,
        mother_pool=mother_test,
        father_indices_global=father_test_idx,
        mother_indices_global=mother_test_idx,
        output_dir=output_dir,
        prefix="result",
        seed=sample_seed,
        n_show=args.n_show,
        n_samples=args.n_final_samples,
    )


if __name__ == "__main__":
    main()
