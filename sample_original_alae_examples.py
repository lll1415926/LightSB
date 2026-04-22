import argparse
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "ALAE"))

from src.light_sb import LightSB
from alae_ffhq_inference import decode, load_model


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample extra examples from an existing original LightSB ALAE checkpoint."
    )
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--input-data", type=str, default="MAN")
    p.add_argument("--target-data", type=str, default="WOMAN")
    p.add_argument("--young-min-age", type=int, default=18)
    p.add_argument("--young-max-age", type=int, default=30)
    p.add_argument("--old-min-age", type=int, default=55)
    p.add_argument("--old-max-age", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--n-potentials", type=int, default=10)
    p.add_argument("--n-show", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--output-name", type=str, default=None)
    return p.parse_args()


def get_inds(g, a, label, young_min_age, young_max_age, old_min_age, old_max_age):
    g = g.reshape(-1)
    a = a.reshape(-1)

    valid_age = a != -1
    male = g == "male"
    female = g == "female"

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
        return np.where(valid_age & (a >= young_min_age) & (a < young_max_age))[0]
    if label == "OLD":
        return np.where(valid_age & (a >= old_min_age) & (a < old_max_age))[0]
    if label == "YOUNG_MAN":
        return np.where(
            male & valid_age & (a >= young_min_age) & (a < young_max_age)
        )[0]
    if label == "OLD_MAN":
        return np.where(
            male & valid_age & (a >= old_min_age) & (a < old_max_age)
        )[0]
    if label == "YOUNG_WOMAN":
        return np.where(
            female & valid_age & (a >= young_min_age) & (a < young_max_age)
        )[0]
    if label == "OLD_WOMAN":
        return np.where(
            female & valid_age & (a >= old_min_age) & (a < old_max_age)
        )[0]
    raise ValueError(f"Unknown label: {label}")


def main():
    args = parse_args()

    ckpt_dir = os.path.join(ROOT, "checkpoints", args.exp_name)
    ckpt_path = os.path.join(ckpt_dir, "D.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_dir = os.path.join(ROOT, "data")
    train_size = 60000

    latents = np.load(os.path.join(data_dir, "latents.npy"))
    gender = np.load(os.path.join(data_dir, "gender.npy"))
    age = np.load(os.path.join(data_dir, "age.npy"))
    test_inp_imgs = np.load(os.path.join(data_dir, "test_images.npy"))

    test_latents = latents[train_size:]
    test_gender = gender[train_size:]
    test_age = age[train_size:]

    x_inds_test = get_inds(
        test_gender,
        test_age,
        args.input_data,
        args.young_min_age,
        args.young_max_age,
        args.old_min_age,
        args.old_max_age,
    )
    if len(x_inds_test) < args.n_show:
        raise ValueError(
            f"Need at least {args.n_show} test samples for {args.input_data}, got {len(x_inds_test)}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = LightSB(
        dim=512,
        n_potentials=args.n_potentials,
        epsilon=args.epsilon,
        sampling_batch_size=128,
        S_diagonal_init=0.1,
        is_diagonal=True,
    ).cpu()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    alae_cfg = os.path.join(ROOT, "ALAE", "configs", "ffhq.yaml")
    alae_dir = os.path.join(ROOT, "ALAE", "training_artifacts", "ffhq")
    os.chdir(os.path.join(ROOT, "ALAE"))
    alae_model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
    os.chdir(ROOT)

    inds_to_map = np.random.choice(len(x_inds_test), size=args.n_show, replace=False)
    chosen = x_inds_test[inds_to_map]
    latent_to_map = torch.tensor(test_latents[chosen], dtype=torch.float32)
    inp_images = test_inp_imgs[chosen]

    with torch.no_grad():
        mapped_all = [model(latent_to_map) for _ in range(args.n_samples)]
        mapped = torch.stack(mapped_all, dim=1)

    decoded_all = []
    with torch.no_grad():
        for k in range(args.n_samples):
            img = decode(alae_model, mapped[:, k])
            img = (
                ((img * 0.5 + 0.5) * 255)
                .clamp(0, 255)
                .byte()
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            decoded_all.append(img)
    decoded_all = np.stack(decoded_all, axis=1)

    fig, axes = plt.subplots(
        args.n_show, args.n_samples + 1, figsize=(args.n_samples + 1, args.n_show), dpi=200
    )
    for i in range(args.n_show):
        axes[i][0].imshow(inp_images[i])
        axes[i][0].axis("off")
        for k in range(args.n_samples):
            axes[i][k + 1].imshow(decoded_all[i, k])
            axes[i][k + 1].axis("off")

    axes[0][0].set_title("Input", fontsize=5)
    for k in range(args.n_samples):
        axes[0][k + 1].set_title(f"Sample {k + 1}", fontsize=5)

    fig.suptitle(f"{args.input_data} -> {args.target_data}", fontsize=7, y=1.01)
    fig.tight_layout(pad=0.05)

    if args.output_name is None:
        output_name = f"result_seed_{args.seed}.png"
    else:
        output_name = args.output_name
    out_fig = os.path.join(ckpt_dir, output_name)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print(out_fig)


if __name__ == "__main__":
    main()
