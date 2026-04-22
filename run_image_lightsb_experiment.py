"""
Prototype image-space LightSB experiment.

This keeps the LightSB training shape (-log_potential(y) + log_C(x)) but
replaces the closed-form Gaussian bridge with a convolutional residual bridge
that operates directly on images.

It prefers a pre-decoded image cache such as `data/decoded_images_64.npy`.
If no cache is found for the requested resolution, it falls back to the small
`test_images.npy` subset.
"""
import argparse
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.image_space_light_sb import ImageSpaceLightSB
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
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--noise-std", type=float, default=0.04)
    p.add_argument("--residual-scale", type=float, default=0.35)
    p.add_argument("--lambda-transport", type=float, default=2.0)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--output-seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--exp-suffix", type=str, default="")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--image-cache", type=str, default=None)
    return p.parse_args()


def get_inds(g, a, label, young_min_age, young_max_age, old_min_age, old_max_age):
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
        return np.where(valid_age & (a >= young_min_age) & (a < young_max_age))[0]
    if label == "OLD":
        return np.where(valid_age & (a >= old_min_age) & (a < old_max_age))[0]
    if label == "YOUNG_MAN":
        return np.where(male & valid_age & (a >= young_min_age) & (a < young_max_age))[0]
    if label == "OLD_MAN":
        return np.where(male & valid_age & (a >= old_min_age) & (a < old_max_age))[0]
    if label == "YOUNG_WOMAN":
        return np.where(female & valid_age & (a >= young_min_age) & (a < young_max_age))[0]
    if label == "OLD_WOMAN":
        return np.where(female & valid_age & (a >= old_min_age) & (a < old_max_age))[0]
    raise ValueError(f"Unknown label: {label}")


def preprocess_images(images, image_size):
    x = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0
    x = x * 2.0 - 1.0
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x


def to_uint8(x):
    return ((x.clamp(-1, 1) * 0.5 + 0.5) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def load_image_data(data_dir, image_size, cache_override=None):
    if cache_override is not None:
        cache_path = cache_override
    else:
        cache_path = os.path.join(data_dir, f"decoded_images_{image_size}.npy")

    gender = np.load(os.path.join(data_dir, "gender.npy"))
    age = np.load(os.path.join(data_dir, "age.npy"))

    if os.path.exists(cache_path):
        images = np.load(cache_path, mmap_mode="r")
        print(f"Using decoded image cache: {cache_path}")
        return images, gender, age

    print("Decoded image cache not found, falling back to test_images.npy")
    images = np.load(os.path.join(data_dir, "test_images.npy"))
    return images, gender[60000:60300], age[60000:60300]


def main():
    args = parse_args()
    torch.manual_seed(args.output_seed)
    np.random.seed(args.output_seed)

    exp_name = (
        f"ImageLightSB_{args.input_data}_TO_{args.target_data}"
        f"_IMG{args.image_size}_EPS_{args.epsilon}"
    )
    if args.exp_suffix:
        exp_name = f"{exp_name}_{args.exp_suffix.strip()}"
    output_path = os.path.join(ROOT, "checkpoints", exp_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"=== {exp_name} ===")
    print(f"Output: {output_path}")

    data_dir = os.path.join(ROOT, "data")
    all_images, gender, age = load_image_data(data_dir, args.image_size, args.image_cache)

    x_inds = get_inds(
        gender, age, args.input_data,
        args.young_min_age, args.young_max_age, args.old_min_age, args.old_max_age
    )
    y_inds = get_inds(
        gender, age, args.target_data,
        args.young_min_age, args.young_max_age, args.old_min_age, args.old_max_age
    )
    assert len(x_inds) > 0, f"No source images for {args.input_data}"
    assert len(y_inds) > 0, f"No target images for {args.target_data}"

    x_all = preprocess_images(np.asarray(all_images[x_inds]), args.image_size)
    y_all = preprocess_images(np.asarray(all_images[y_inds]), args.image_size)

    print(f"Source ({args.input_data}) images: {len(x_all)}")
    print(f"Target ({args.target_data}) images: {len(y_all)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_sampler = TensorSampler(x_all, device=device)
    y_sampler = TensorSampler(y_all, device=device)

    model = ImageSpaceLightSB(
        image_channels=3,
        n_components=args.n_components,
        feature_dim=args.feature_dim,
        base_channels=args.base_channels,
        epsilon=args.epsilon,
        noise_std=args.noise_std,
        residual_scale=args.residual_scale,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = os.path.join(output_path, "D.pt")
    loss_curve_path = os.path.join(output_path, "loss.npy")
    if os.path.exists(ckpt_path) and not args.force_retrain:
        print("Found checkpoint, loading and skipping training.")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        losses = np.load(loss_curve_path) if os.path.exists(loss_curve_path) else None
    else:
        losses = []
        for step in tqdm(range(args.max_steps)):
            opt.zero_grad()
            x0 = x_sampler.sample(min(args.batch_size, len(x_all)))
            y1 = y_sampler.sample(min(args.batch_size, len(y_all)))
            objective = model.training_loss(x0, y1)
            transport_reg = model.transport_regularization(x0)
            loss = objective + args.lambda_transport * transport_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))
            if (step + 1) % args.print_every == 0 or step == 0:
                print(
                    f"step={step + 1:04d} "
                    f"loss={loss.item():.4f} "
                    f"objective={objective.item():.4f} "
                    f"transport={transport_reg.item():.4f}"
                )

        torch.save(model.state_dict(), ckpt_path)
        np.save(loss_curve_path, np.array(losses))

        plt.figure(figsize=(6, 3), dpi=150)
        plt.plot(losses)
        plt.title("Image-space LightSB loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "loss.png"))
        plt.close()

    with torch.no_grad():
        n_show = min(8, len(x_all))
        show_x = x_all[:n_show].to(device)
        mapped = [model(show_x, deterministic=False) for _ in range(3)]
        mapped = torch.stack(mapped, dim=1)

    inp_imgs = to_uint8(show_x)
    out_imgs = to_uint8(mapped.reshape(-1, 3, args.image_size, args.image_size)).reshape(
        n_show, 3, args.image_size, args.image_size, 3
    )

    fig, axes = plt.subplots(n_show, 4, figsize=(4, n_show), dpi=180)
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(n_show):
        axes[i][0].imshow(inp_imgs[i])
        axes[i][0].axis("off")
        for k in range(3):
            axes[i][k + 1].imshow(out_imgs[i, k])
            axes[i][k + 1].axis("off")
    axes[0][0].set_title("Input", fontsize=6)
    for k in range(3):
        axes[0][k + 1].set_title(f"Sample {k + 1}", fontsize=6)
    fig.tight_layout(pad=0.05)
    fig.savefig(os.path.join(output_path, "result.png"), bbox_inches="tight")
    plt.close()

    print(f"Result saved: {os.path.join(output_path, 'result.png')}")


if __name__ == "__main__":
    main()
