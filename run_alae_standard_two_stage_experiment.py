"""
Two-stage standard LightSB ALAE image translation experiment.
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
from src.distributions import TensorSampler
from src.light_sb import LightSB


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
    p.add_argument("--n-potentials", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--sampling-batch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--d-lr", type=float, default=1e-3)
    p.add_argument("--output-seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-show", type=int, default=8)
    p.add_argument("--n-final-samples", type=int, default=3)
    p.add_argument("--print-every", type=int, default=200)
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


def stage_name(src, tgt):
    return f"LightSB_ALAE_{src}_TO_{tgt}_EPSILON_{args.epsilon}"


def build_model(device):
    return LightSB(
        dim=args.dim,
        n_potentials=args.n_potentials,
        epsilon=args.epsilon,
        is_diagonal=True,
        sampling_batch_size=args.sampling_batch_size,
    ).to(device)


def train_or_load_stage(src, tgt, train_latents, train_gender, train_age, device):
    ckpt_dir = os.path.join(ROOT, "checkpoints", stage_name(src, tgt))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "D.pt")

    model = build_model(device)
    x_inds_train = get_inds(train_gender, train_age, src)
    y_inds_train = get_inds(train_gender, train_age, tgt)

    X_train = torch.tensor(train_latents[x_inds_train], dtype=torch.float32)
    Y_train = torch.tensor(train_latents[y_inds_train], dtype=torch.float32)

    assert len(X_train) > 0, f"No source train samples for {src}"
    assert len(Y_train) > 0, f"No target train samples for {tgt}"

    if os.path.exists(ckpt_path):
        print(f"Found checkpoint for {src} -> {tgt}, loading.")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        return model

    print(f"Training standard LightSB: {src} -> {tgt}")
    X_sampler = TensorSampler(X_train, device=device)
    Y_sampler = TensorSampler(Y_train, device=device)

    with torch.no_grad():
        model.init_r_by_samples(Y_sampler.sample(args.n_potentials))
        model.log_alpha_raw.data.zero_()

    opt = torch.optim.Adam(model.parameters(), lr=args.d_lr)

    losses = []
    for step in tqdm(range(args.max_steps)):
        opt.zero_grad()
        X0 = X_sampler.sample(args.batch_size)
        X1 = Y_sampler.sample(args.batch_size)

        loss = (-model.get_log_potential(X1) + model.get_log_C(X0)).mean()
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
        if step % args.print_every == 0 or step == args.max_steps - 1:
            print(f"[{src} -> {tgt}] step {step:5d} | loss={loss.item():+.6f}")

    torch.save(model.state_dict(), ckpt_path)
    np.save(os.path.join(ckpt_dir, "D_loss.npy"), np.array(losses))
    plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(losses)
    plt.title(f"{src} -> {tgt} loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, "D_loss.png"))
    plt.close()

    model.eval()
    return model


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


def main():
    device = torch.device(args.device)
    torch.manual_seed(args.output_seed)
    np.random.seed(args.output_seed)

    latents = np.load(os.path.join(ROOT, "data", "latents.npy"))
    gender = np.load(os.path.join(ROOT, "data", "gender.npy"))
    age = np.load(os.path.join(ROOT, "data", "age.npy"))
    test_inp_imgs = np.load(os.path.join(ROOT, "data", "test_images.npy"))

    train_latents, test_latents = latents[:TRAIN_SIZE], latents[TRAIN_SIZE:]
    train_gender, test_gender = gender[:TRAIN_SIZE], gender[TRAIN_SIZE:]
    train_age, test_age = age[:TRAIN_SIZE], age[TRAIN_SIZE:]

    stage1 = train_or_load_stage(
        args.input_data, args.middle_data, train_latents, train_gender, train_age, device
    )
    stage2 = train_or_load_stage(
        args.middle_data, args.target_data, train_latents, train_gender, train_age, device
    )

    x_inds_test = get_inds(test_gender, test_age, args.input_data)
    valid_pos = np.where(x_inds_test < len(test_inp_imgs))[0]
    assert len(valid_pos) > 0, "No valid source test images found."

    n_show = min(args.n_show, len(valid_pos))
    inds_to_map = np.random.choice(valid_pos, size=n_show, replace=False)
    lat_idx = x_inds_test[inds_to_map]

    latent_to_map = torch.tensor(test_latents[lat_idx], dtype=torch.float32, device=device)
    inp_images = test_inp_imgs[lat_idx]

    with torch.no_grad():
        stage1_mid = stage1(latent_to_map)
        stage2_final = [stage2(stage1_mid) for _ in range(args.n_final_samples)]
    stage2_final = torch.stack(stage2_final, dim=1)

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

    exp_name = (
        f"LightSB_ALAE_{args.input_data}_TO_{args.middle_data}_TO_{args.target_data}"
        f"_EPSILON_{args.epsilon}"
    )
    if args.exp_suffix.strip():
        exp_name = f"{exp_name}_{args.exp_suffix.strip()}"
    output_dir = os.path.join(ROOT, "checkpoints", exp_name)
    os.makedirs(output_dir, exist_ok=True)

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


if __name__ == "__main__":
    main()
