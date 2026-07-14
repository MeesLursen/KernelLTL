"""Build the three embedding-ablation datasets for the G1b feasibility floor.

Creates, as siblings of the validation dir, ``validation_ablation_{zero,mean,shuffle}``
-- each a copy of the validation dataset with a CORRUPTED ``embeddings.pt`` and a relative
``satisfactions.pt`` symlink back to ``../<validation>/satisfactions.pt`` (so the satvecs are
never triplicated). ``validation_ablation.py`` then conditions on these as-is (no in-pipeline
ablation), matching the structure the runner + snellius_validate_ablation.sh expect.

Corruption (every row, mirroring ``validation_utils._apply_embedding_ablation``):
  zero    -- every embedding -> 0 (the unconditional prior).
  mean    -- every embedding -> the dataset-mean embedding (constant, target-agnostic).
  shuffle -- the embeddings globally permuted across targets (seeded): a real, mismatched signal.

The depth-graded validation set contains no trivial (tautology/contradiction) targets by
construction, so corruption applies uniformly to all rows.

Usage:
    python scripts/build_validation_ablation.py --validation-dir <.../datasets/validation> --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

MODES = ("zero", "mean", "shuffle")


def _corrupt(emb: torch.Tensor, mode: str, gen: torch.Generator) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(emb)
    if mode == "mean":
        return emb.mean(dim=0, keepdim=True).expand_as(emb).contiguous()
    if mode == "shuffle":
        if emb.size(0) < 2:
            return emb.clone()
        return emb[torch.randperm(emb.size(0), generator=gen)].contiguous()
    raise ValueError(f"unknown mode {mode!r}")


def _write_ablation_dir(validation_dir: str, out_dir: str, corrupted: torch.Tensor, mode: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(corrupted, os.path.join(out_dir, "embeddings.pt"))
    shutil.copy2(os.path.join(validation_dir, "formulas.jsonl"), os.path.join(out_dir, "formulas.jsonl"))

    with open(os.path.join(validation_dir, "metadata.json")) as fp:
        meta = json.load(fp)
    meta.setdefault("extra_metadata", {})
    meta["extra_metadata"]["embedding_ablation"] = mode
    meta["has_satisfactions"] = True
    meta["store_satisfaction"] = True
    with open(os.path.join(out_dir, "metadata.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    # satisfactions: relative symlink to the shared validation satvecs (never triplicated)
    link = os.path.join(out_dir, "satisfactions.pt")
    target = os.path.join("..", os.path.basename(os.path.normpath(validation_dir)), "satisfactions.pt")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(target, link)


def parse_args():
    p = argparse.ArgumentParser(description="Build validation_ablation_{zero,mean,shuffle} datasets.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--validation-dir", required=True, help="Path to the validation dataset dir")
    p.add_argument("--output-root", default=None,
                   help="Where the ablation dirs go (default: the validation dir's parent, so the "
                        "../validation satvec symlink resolves)")
    p.add_argument("--seed", type=int, default=0, help="Seed for the shuffle permutation")
    return p.parse_args()


def main():
    args = parse_args()
    validation_dir = os.path.normpath(args.validation_dir)
    if os.path.basename(validation_dir) != "validation":
        raise ValueError(f"validation dir must be named 'validation' (got {os.path.basename(validation_dir)}) "
                         "so the ../validation satisfactions symlink resolves.")
    out_root = args.output_root or os.path.dirname(validation_dir)
    if os.path.dirname(validation_dir) != os.path.normpath(out_root):
        raise ValueError("output-root must be the validation dir's parent so ../validation resolves.")

    emb = torch.load(os.path.join(validation_dir, "embeddings.pt"), map_location="cpu").to(torch.float32)
    if emb.ndim != 2:
        raise ValueError(f"expected 2-D embeddings, got shape {tuple(emb.shape)}")
    print(f"[ablation-build] {emb.shape[0]} targets")

    gen = torch.Generator(); gen.manual_seed(int(args.seed))
    for mode in MODES:
        corrupted = _corrupt(emb, mode, gen)
        out_dir = os.path.join(out_root, f"validation_ablation_{mode}")
        _write_ablation_dir(validation_dir, out_dir, corrupted, mode)
        print(f"[ablation-build] wrote {out_dir}  (embeddings.pt {tuple(corrupted.shape)}, "
              f"satisfactions.pt -> {os.path.join('..', os.path.basename(validation_dir), 'satisfactions.pt')})")
    print("[ablation-build] done.")


if __name__ == "__main__":
    main()
