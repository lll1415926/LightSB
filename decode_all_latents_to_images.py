import argparse
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "ALAE"))

from alae_ffhq_inference import decode, load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def decode_batch(model, z, resolution):
    with torch.no_grad():
        out = decode(model, z)
        if out.shape[-1] != resolution or out.shape[-2] != resolution:
            out = F.interpolate(
                out,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )
    return out


def main():
    args = parse_args()

    latents = np.load(os.path.join(ROOT, "data", "latents.npy"))
    total = latents.shape[0]
    start = max(0, args.start)
    end = total if args.end is None else min(total, args.end)
    assert start < end <= total

    if args.output is None:
        if start == 0 and end == total:
            output_name = f"decoded_images_{args.resolution}.npy"
        else:
            output_name = f"decoded_images_{args.resolution}_{start}_{end}.npy"
    else:
        output_name = args.output

    output_path = os.path.join(ROOT, "data", output_name)
    if os.path.exists(output_path):
        os.remove(output_path)

    alae_cfg = os.path.join(ROOT, "ALAE", "configs", "ffhq.yaml")
    alae_dir = os.path.join(ROOT, "ALAE", "training_artifacts", "ffhq")
    os.chdir(os.path.join(ROOT, "ALAE"))
    model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
    os.chdir(ROOT)

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model = model.to(device)

    print(
        f"Decoding latents[{start}:{end}] with standard ALAE decode "
        f"then resizing to {args.resolution} on {device}."
    )
    print(f"Saving to {output_path}")

    num = end - start
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint8,
        shape=(num, args.resolution, args.resolution, 3),
    )

    for offset in tqdm(range(0, num, args.batch_size)):
        batch_start = start + offset
        batch_end = min(end, batch_start + args.batch_size)
        z = torch.tensor(latents[batch_start:batch_end], dtype=torch.float32, device=device)
        decoded = decode_batch(model, z, args.resolution)
        decoded = (
            ((decoded * 0.5 + 0.5) * 255)
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        out[offset:offset + len(decoded)] = decoded
        out.flush()

    print(output_path)


if __name__ == "__main__":
    main()
