"""
Prepare TSKinFace triplets for the static parent-child LightSB experiment.

This script reads TSKinFace family triples from either:
1. a manifest CSV/TSV with explicit father/mother/child image paths, or
2. an auto-discovered directory tree with one family per leaf directory.

It then uses the existing ALAE encoder to produce a latent dataset:
    z_f:   [N, d]
    z_m:   [N, d]
    y:     [N, d]
    a:     [N]    child-condition label (0=son, 1=daughter by default)
    split: [N]    "train" or "test"
    relation: [N] "FM-S" or "FM-D"
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "ALAE"))

from alae_ffhq_inference import encode, load_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FamilyTriple:
    father_path: str
    mother_path: str
    child_path: str
    relation: str
    family_id: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--output-path", type=str, default=os.path.join("data", "tskinface_latents.npz"))
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--manifest-delimiter", type=str, default="")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-families", type=int, default=0)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--num-folds", type=int, default=0)
    p.add_argument("--fold-index", type=int, default=0)
    p.add_argument("--seed", type=lambda x: int(x, 0), default=0xBADBEEF)
    p.add_argument("--alae-config", type=str, default=os.path.join("ALAE", "configs", "ffhq.yaml"))
    p.add_argument("--alae-artifacts", type=str, default=os.path.join("ALAE", "training_artifacts", "ffhq"))
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _normalize_relation(value: str) -> str:
    raw = value.strip().lower().replace("_", "-")
    if raw in {"fm-s", "fms", "father-mother-son", "son"}:
        return "FM-S"
    if raw in {"fm-d", "fmd", "father-mother-daughter", "daughter"}:
        return "FM-D"
    return value.strip()


def _relation_to_label(relation: str) -> int:
    relation = _normalize_relation(relation)
    if relation == "FM-S":
        return 0
    if relation == "FM-D":
        return 1
    raise ValueError(f"Unsupported relation '{relation}'. Expected FM-S or FM-D.")


def _guess_delimiter(path: str, preferred: str) -> str:
    if preferred:
        return preferred
    if path.lower().endswith(".tsv"):
        return "\t"
    return ","


def _resolve_path(path_value: str, root: str) -> str:
    path_value = path_value.strip()
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(root, path_value)


def load_manifest(manifest_path: str, dataset_root: str, delimiter: str) -> List[FamilyTriple]:
    triples: List[FamilyTriple] = []
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        required = {"father", "mother", "child"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing)}. "
                f"Required columns are father,mother,child and optional relation,family_id."
            )
        for idx, row in enumerate(reader):
            relation = _normalize_relation(row.get("relation", "").strip())
            if not relation:
                child_lower = os.path.basename(row["child"]).lower()
                if "daughter" in child_lower or "_d" in child_lower or "-d" in child_lower:
                    relation = "FM-D"
                else:
                    relation = "FM-S"
            triples.append(
                FamilyTriple(
                    father_path=_resolve_path(row["father"], dataset_root),
                    mother_path=_resolve_path(row["mother"], dataset_root),
                    child_path=_resolve_path(row["child"], dataset_root),
                    relation=relation,
                    family_id=row.get("family_id", f"family_{idx:05d}").strip() or f"family_{idx:05d}",
                )
            )
    return triples


def _list_image_files(path: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(full) and ext in IMAGE_EXTENSIONS:
            files.append(full)
    return files


def _infer_role_from_name(path: str) -> Optional[str]:
    name = os.path.basename(path).lower()
    stem = os.path.splitext(name)[0]
    tokens = set(
        stem.replace("-", "_").replace(" ", "_").split("_")
    )

    if "father" in tokens or "dad" in tokens or tokens.intersection({"f", "fa"}):
        return "father"
    if "mother" in tokens or "mom" in tokens or "mum" in tokens or tokens.intersection({"m", "mo"}):
        return "mother"
    if "son" in tokens or tokens.intersection({"s", "child", "kid"}):
        return "child_son"
    if "daughter" in tokens or tokens.intersection({"d", "girl"}):
        return "child_daughter"

    if "father" in stem or "dad" in stem:
        return "father"
    if "mother" in stem or "mom" in stem or "mum" in stem:
        return "mother"
    if "son" in stem:
        return "child_son"
    if "daughter" in stem:
        return "child_daughter"
    return None


def _infer_relation_from_path(path: str) -> Optional[str]:
    lowered = path.lower().replace("_", "-")
    if "fm-s" in lowered or "fms" in lowered:
        return "FM-S"
    if "fm-d" in lowered or "fmd" in lowered:
        return "FM-D"
    return None


def auto_discover_triples(dataset_root: str) -> List[FamilyTriple]:
    cropped_root = os.path.join(dataset_root, "TSKinFace_Data", "TSKinFace_cropped")
    if os.path.isdir(cropped_root):
        triples = discover_tskinface_cropped(cropped_root)
        if triples:
            return triples

    triples: List[FamilyTriple] = []
    for current_root, dirs, _files in os.walk(dataset_root):
        if dirs:
            continue
        image_files = _list_image_files(current_root)
        if len(image_files) < 3:
            continue

        father_path = None
        mother_path = None
        child_path = None
        child_role = None

        for path in image_files:
            role = _infer_role_from_name(path)
            if role == "father" and father_path is None:
                father_path = path
            elif role == "mother" and mother_path is None:
                mother_path = path
            elif role in {"child_son", "child_daughter"} and child_path is None:
                child_path = path
                child_role = role

        if father_path is None or mother_path is None or child_path is None:
            continue

        relation = _infer_relation_from_path(current_root)
        if relation is None:
            if child_role == "child_son":
                relation = "FM-S"
            elif child_role == "child_daughter":
                relation = "FM-D"
            else:
                continue

        triples.append(
            FamilyTriple(
                father_path=father_path,
                mother_path=mother_path,
                child_path=child_path,
                relation=relation,
                family_id=os.path.relpath(current_root, dataset_root).replace("\\", "/"),
            )
        )
    return triples


def discover_tskinface_cropped(cropped_root: str) -> List[FamilyTriple]:
    triples: List[FamilyTriple] = []
    relation_dirs = [("FMS", "FM-S"), ("FMD", "FM-D")]

    for dir_name, relation in relation_dirs:
        relation_dir = os.path.join(cropped_root, dir_name)
        if not os.path.isdir(relation_dir):
            continue

        grouped: Dict[str, Dict[str, str]] = {}
        for name in sorted(os.listdir(relation_dir)):
            full = os.path.join(relation_dir, name)
            if not os.path.isfile(full):
                continue
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            parts = stem.split("-")
            if len(parts) < 3:
                continue
            family_key = "-".join(parts[:-1])
            role = parts[-1].upper()
            grouped.setdefault(family_key, {})[role] = full

        for family_key, roles in sorted(grouped.items()):
            father_path = roles.get("F")
            mother_path = roles.get("M")
            child_path = roles.get("S") if relation == "FM-S" else roles.get("D")
            if father_path is None or mother_path is None or child_path is None:
                continue

            triples.append(
                FamilyTriple(
                    father_path=father_path,
                    mother_path=mother_path,
                    child_path=child_path,
                    relation=relation,
                    family_id=family_key,
                )
            )

    return triples


def load_image_as_tensor(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)
    x = x / 127.5 - 1.0
    if x.shape[1] != image_size or x.shape[2] != image_size:
        x = F.interpolate(
            x.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return x


def batched(items: Sequence[FamilyTriple], batch_size: int) -> Iterable[Sequence[FamilyTriple]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def encode_images_to_latents(
    model,
    image_paths: Sequence[str],
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    tensors = [load_image_as_tensor(path, image_size) for path in image_paths]
    batch = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        z = encode(model, batch)
        if z.ndim == 3:
            z = z[:, 0, :]
    return z.detach().cpu().numpy().astype(np.float32)


def build_splits(n: int, test_ratio: float, num_folds: int, fold_index: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    split = np.empty(n, dtype=object)
    split[:] = "train"

    if num_folds > 1:
        fold_index = fold_index % num_folds
        fold_sizes = np.full(num_folds, n // num_folds, dtype=np.int64)
        fold_sizes[: n % num_folds] += 1
        start = int(fold_sizes[:fold_index].sum())
        end = start + int(fold_sizes[fold_index])
        test_idx = indices[start:end]
    else:
        test_count = max(1, int(round(n * test_ratio)))
        test_idx = indices[:test_count]

    split[test_idx] = "test"
    return split


def main():
    args = parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.manifest:
        manifest_path = os.path.abspath(args.manifest)
        delimiter = _guess_delimiter(manifest_path, args.manifest_delimiter)
        triples = load_manifest(manifest_path, dataset_root, delimiter)
        source_desc = f"manifest {manifest_path}"
    else:
        triples = auto_discover_triples(dataset_root)
        source_desc = f"auto-discovery under {dataset_root}"

    if args.max_families > 0:
        triples = triples[: args.max_families]

    if not triples:
        raise ValueError(
            "No family triples were found. "
            "Provide a manifest with father,mother,child columns or organize the dataset "
            "so that each leaf directory contains clearly named father/mother/son|daughter images."
        )

    split = build_splits(
        n=len(triples),
        test_ratio=args.test_ratio,
        num_folds=args.num_folds,
        fold_index=args.fold_index,
        seed=args.seed,
    )

    print(f"Found {len(triples)} family triples from {source_desc}.")
    print(
        f"Split mode: {'fold' if args.num_folds > 1 else 'random'} | "
        f"train={(split == 'train').sum()} test={(split == 'test').sum()}"
    )

    device = choose_device(args.device)
    print(f"Loading ALAE model on {device}...")

    alae_cfg = os.path.join(ROOT, args.alae_config)
    alae_dir = os.path.join(ROOT, args.alae_artifacts)
    os.chdir(os.path.join(ROOT, "ALAE"))
    model = load_model(alae_cfg, training_artifacts_dir=alae_dir)
    os.chdir(ROOT)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    z_f_all = []
    z_m_all = []
    y_all = []
    a_all = []
    relation_all = []
    family_id_all = []

    for batch in tqdm(list(batched(triples, args.batch_size)), desc="Encoding families"):
        father_paths = [item.father_path for item in batch]
        mother_paths = [item.mother_path for item in batch]
        child_paths = [item.child_path for item in batch]

        zf = encode_images_to_latents(model, father_paths, args.image_size, device)
        zm = encode_images_to_latents(model, mother_paths, args.image_size, device)
        yc = encode_images_to_latents(model, child_paths, args.image_size, device)

        z_f_all.append(zf)
        z_m_all.append(zm)
        y_all.append(yc)
        a_all.extend(_relation_to_label(item.relation) for item in batch)
        relation_all.extend(_normalize_relation(item.relation) for item in batch)
        family_id_all.extend(item.family_id for item in batch)

    z_f = np.concatenate(z_f_all, axis=0)
    z_m = np.concatenate(z_m_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    a = np.asarray(a_all, dtype=np.int64)
    relation = np.asarray(relation_all, dtype=object)
    family_id = np.asarray(family_id_all, dtype=object)

    np.savez(
        output_path,
        z_f=z_f,
        z_m=z_m,
        y=y,
        a=a,
        split=split,
        relation=relation,
        family_id=family_id,
    )
    print(f"Saved latent triplets to {output_path}")
    print(f"Shapes: z_f={z_f.shape}, z_m={z_m.shape}, y={y.shape}, a={a.shape}")


if __name__ == "__main__":
    main()
