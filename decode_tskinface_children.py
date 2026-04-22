"""
Decode child samples from a trained static parent-child LightSB checkpoint.

Given:
1. a trained `StaticParentChildLightSB` checkpoint,
2. a prepared `tskinface_latents.npz`,
3. the downloaded TSKinFace image directory,

this script samples child latents for a subset of families and decodes them
with the existing ALAE decoder. It saves both a latent archive and a preview
grid that shows:

    father | mother | true child | sample 1 | sample 2 | ...
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "ALAE"))

from alae_ffhq_inference import decode, load_model
from src.conditional_static_lightsb import StaticParentChildLightSB


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--n-potentials", type=int, default=4)
    p.add_argument("--n-conditions", type=int, default=2)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--use-interaction", action="store_true")
    p.add_argument("--num-families", type=int, default=8)
    p.add_argument("--samples-per-family", type=int, default=3)
    p.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    p.add_argument("--seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--decode-device", type=str, default="cpu")
    p.add_argument("--decode-size", type=int, default=256)
    p.add_argument("--alae-config", type=str, default=os.path.join("ALAE", "configs", "ffhq.yaml"))
    p.add_argument("--alae-artifacts", type=str, default=os.path.join("ALAE", "training_artifacts", "ffhq"))
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def decode_batch(model, z: torch.Tensor, image_size: int) -> torch.Tensor:
    with torch.no_grad():
        out = decode(model, z)
        if out.shape[-1] != image_size or out.shape[-2] != image_size:
            out = F.interpolate(
                out,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
    return out


def to_uint8(x: torch.Tensor) -> np.ndarray:
    return (
        ((x.clamp(-1, 1) * 0.5 + 0.5) * 255)
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def load_face_image(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def family_to_paths(dataset_root: str, family_id: str, relation: str):
    cropped_root = os.path.join(dataset_root, "TSKinFace_Data", "TSKinFace_cropped")
    relation_dir = "FMS" if relation == "FM-S" else "FMD"
    child_suffix = "S" if relation == "FM-S" else "D"
    rel_dir = os.path.join(cropped_root, relation_dir)

    father = os.path.join(rel_dir, f"{family_id}-F.jpg")
    mother = os.path.join(rel_dir, f"{family_id}-M.jpg")
    child = os.path.join(rel_dir, f"{family_id}-{child_suffix}.jpg")
    return father, mother, child


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    data = np.load(args.latent_path, allow_pickle=True)
    split = data["split"].astype(str)
    if args.split == "all":
        mask = np.ones(len(split), dtype=bool)
    else:
        mask = split == args.split

    z_f = torch.tensor(data["z_f"][mask], dtype=torch.float32, device=device)
    z_m = torch.tensor(data["z_m"][mask], dtype=torch.float32, device=device)
    y_true = torch.tensor(data["y"][mask], dtype=torch.float32, device=device)
    a = torch.tensor(data["a"][mask], dtype=torch.long, device=device)
    family_ids = data["family_id"][mask].astype(str)
    relations = data["relation"][mask].astype(str)

    n_available = z_f.shape[0]
    n_show = min(args.num_families, n_available)
    if n_show == 0:
        raise ValueError("No families available for the requested split.")

    selected = np.random.choice(n_available, size=n_show, replace=False)

    model = StaticParentChildLightSB(
        latent_dim=args.latent_dim,
        n_components=args.n_potentials,
        epsilon=args.epsilon,
        num_conditions=args.n_conditions,
        use_interaction=args.use_interaction,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    zf_sel = z_f[selected]
    zm_sel = z_m[selected]
    yt_sel = y_true[selected]
    a_sel = a[selected]
    family_sel = family_ids[selected]
    relation_sel = relations[selected]

    with torch.no_grad():
        y_samples = model.sample_child(
            zf_sel,
            zm_sel,
            a_sel,
            n_samples=args.samples_per_family,
        )

    decode_device = choose_device(args.decode_device)

    alae_cfg = os.path.join(ROOT, args.alae_config)
    alae_dir = os.path.join(ROOT, args.alae_artifacts)
    os.chdir(os.path.join(ROOT, "ALAE"))
    alae_model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
    os.chdir(ROOT)
    alae_model = alae_model.to(decode_device)
    alae_model.eval()

    yt_sel_decode = yt_sel.to(decode_device)
    y_samples_decode = y_samples.to(decode_device)

    decoded_true = decode_batch(alae_model, yt_sel_decode, args.decode_size)
    decoded_gen = decode_batch(
        alae_model,
        y_samples_decode.reshape(-1, args.latent_dim),
        args.decode_size,
    ).reshape(
        n_show,
        args.samples_per_family,
        3,
        args.decode_size,
        args.decode_size,
    )

    true_imgs = to_uint8(decoded_true)
    gen_imgs = to_uint8(
        decoded_gen.reshape(-1, 3, args.decode_size, args.decode_size)
    ).reshape(
        n_show,
        args.samples_per_family,
        args.decode_size,
        args.decode_size,
        3,
    )

    father_imgs: List[np.ndarray] = []
    mother_imgs: List[np.ndarray] = []
    child_imgs: List[np.ndarray] = []
    for family_id, relation in zip(family_sel, relation_sel):
        father_path, mother_path, child_path = family_to_paths(
            args.dataset_root, family_id, relation
        )
        father_imgs.append(load_face_image(father_path, args.decode_size))
        mother_imgs.append(load_face_image(mother_path, args.decode_size))
        child_imgs.append(load_face_image(child_path, args.decode_size))

    father_imgs = np.stack(father_imgs, axis=0)
    mother_imgs = np.stack(mother_imgs, axis=0)
    child_imgs = np.stack(child_imgs, axis=0)

    np.savez(
        os.path.join(args.output_dir, "generated_children.npz"),
        family_id=family_sel,
        relation=relation_sel,
        z_f=zf_sel.cpu().numpy(),
        z_m=zm_sel.cpu().numpy(),
        y_true=yt_sel.cpu().numpy(),
        y_samples=y_samples.cpu().numpy(),
    )

    n_cols = 3 + args.samples_per_family
    fig, axes = plt.subplots(
        n_show,
        n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_show),
        dpi=180,
    )
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_show):
        axes[i, 0].imshow(father_imgs[i])
        axes[i, 1].imshow(mother_imgs[i])
        axes[i, 2].imshow(child_imgs[i])
        axes[i, 0].set_ylabel(f"{family_sel[i]}\n{relation_sel[i]}", fontsize=7)
        for j in range(3):
            axes[i, j].axis("off")
        for k in range(args.samples_per_family):
            axes[i, 3 + k].imshow(gen_imgs[i, k])
            axes[i, 3 + k].axis("off")

    axes[0, 0].set_title("Father", fontsize=8)
    axes[0, 1].set_title("Mother", fontsize=8)
    axes[0, 2].set_title("True Child", fontsize=8)
    for k in range(args.samples_per_family):
        axes[0, 3 + k].set_title(f"Sample {k + 1}", fontsize=8)

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(args.output_dir, "generated_children_grid.png"), bbox_inches="tight")
    plt.close(fig)

    # Also save decoded true/generated-only arrays for downstream use.
    np.savez(
        os.path.join(args.output_dir, "decoded_children.npz"),
        family_id=family_sel,
        relation=relation_sel,
        father=father_imgs,
        mother=mother_imgs,
        true_child=child_imgs,
        decoded_true_child=true_imgs,
        decoded_generated_child=gen_imgs,
    )

    print(os.path.join(args.output_dir, "generated_children_grid.png"))


if __name__ == "__main__":
    main()
