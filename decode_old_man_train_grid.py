import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "ALAE"))

from alae_ffhq_inference import decode, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0xBADBEEF)
    parser.add_argument("--output", type=str, default="old_man_train_decoded_grid.png")
    args = parser.parse_args()

    train_size = 60000
    latents = np.load(os.path.join(ROOT, "data", "latents.npy"))[:train_size]
    age = np.load(os.path.join(ROOT, "data", "age.npy"))[:train_size].reshape(-1)
    gender = np.load(os.path.join(ROOT, "data", "gender.npy"))[:train_size].reshape(-1)

    inds = np.where((gender == "male") & (age != -1) & (age >= 55) & (age < 100))[0]
    assert len(inds) > 0

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(inds, size=min(args.num_samples, len(inds)), replace=False)
    z = torch.tensor(latents[chosen], dtype=torch.float32)

    alae_cfg = os.path.join(ROOT, "ALAE", "configs", "ffhq.yaml")
    alae_dir = os.path.join(ROOT, "ALAE", "training_artifacts", "ffhq")
    os.chdir(os.path.join(ROOT, "ALAE"))
    model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
    os.chdir(ROOT)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    z = z.to(device)
    with torch.no_grad():
        img = decode(model, z)
        img = (
            ((img * 0.5 + 0.5) * 255)
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

    cols = min(8, max(4, int(np.ceil(np.sqrt(len(img))))))
    rows = (len(img) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), dpi=150)
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (idx, im) in enumerate(zip(chosen, img)):
        ax = axes[i // cols, i % cols]
        ax.imshow(im)
        ax.set_title(str(int(idx)), fontsize=6)
        ax.axis("off")

    out = os.path.join(ROOT, args.output)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
