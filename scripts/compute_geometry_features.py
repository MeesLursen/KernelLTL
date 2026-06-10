"""Compute per-target embedding-geometry features for the validation set.

This is the HEAVY step (it touches the ~10 GB satisfactions tensor), intended to
run once on Snellius and cache a small CSV that the light analysis driver
(`visualize_validation_geometry.py`) consumes.

For every validation target phi (keyed by ``formula_id`` = row index into the
saved validation dataset) we compute:

  p            base rate  = mean(satvec)           (informativeness driver)
  variance     p(1-p)     = Bernoulli variance of the satvec
  std          sqrt(variance)
  emb_norm     ||emb(phi)||_2                       (raw conditioning magnitude)
  alignment    || emb(phi) / std_psi ||_2 / std_phi (anchor-coverage; the norm of
               the Pearson-correlation embedding rho_i = cov_i/(std_phi std_psi_i))
  alignment_proxy  emb_norm / std_phi               (cheap fallback, no anchor stds)
  is_trivial   1 if std_phi == 0 (tautology/contradiction; alignment undefined -> 0)

The std/alignment split mirrors the two causes of a small-magnitude embedding
named in the kernel chapter (low variance vs. anchor orthogonality).

Inputs (defaults match the Snellius layout):
  --validation-dataset-dir : holds embeddings.pt (N_t x M) and satisfactions.pt (N_t x N)
  --kernel-dir             : holds F.pt (M x N anchor satisfaction matrix) for std_psi
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validation-dataset-dir", required=True,
                   help="Dir with embeddings.pt and satisfactions.pt (the validation LTLDataset)")
    p.add_argument("--kernel-dir", required=True,
                   help="Dir with F.pt (anchor x trace satisfaction matrix)")
    p.add_argument("--output", required=True, help="Output geometry_features.csv path")
    p.add_argument("--chunk-size", type=int, default=500,
                   help="Rows per chunk when streaming the satisfactions tensor")
    return p.parse_args()


def _load_tensor(path: str, *, mmap: bool = False) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", mmap=mmap) if mmap else \
          torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # be permissive about a few likely keys
        for k in ("embeddings", "satisfactions", "tensor", "data", "F"):
            if k in obj:
                return obj[k]
        raise ValueError(f"{path}: loaded a dict without a recognised tensor key ({list(obj)[:5]})")
    return obj


def _base_rate_streamed(sat_path: str, chunk: int) -> np.ndarray:
    """Mean over the trace axis (dim=1) of a (N_t, N) bool tensor, in row chunks."""
    try:
        sat = _load_tensor(sat_path, mmap=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        _log(f"[geometry] mmap load failed ({exc!r}); falling back to full load")
        sat = _load_tensor(sat_path, mmap=False)
    n_t = sat.shape[0]
    p = np.empty(n_t, dtype=np.float64)
    for i0 in range(0, n_t, chunk):
        i1 = min(n_t, i0 + chunk)
        p[i0:i1] = sat[i0:i1].to(torch.float32).mean(dim=1).numpy()
        if (i0 // chunk) % 10 == 0:
            _log(f"[geometry] base rate {i1}/{n_t}")
    return p


def main() -> None:
    args = parse_args()

    emb_path = os.path.join(args.validation_dataset_dir, "embeddings.pt")
    sat_path = os.path.join(args.validation_dataset_dir, "satisfactions.pt")
    f_path = os.path.join(args.kernel_dir, "F.pt")
    for pth in (emb_path, sat_path, f_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(pth)

    _log("[geometry] loading embeddings ...")
    E = _load_tensor(emb_path).to(torch.float64)                     # (N_t, M)
    n_t, m = E.shape
    emb_norm = E.norm(dim=1).numpy()                                 # (N_t,)
    _log(f"[geometry] embeddings: {n_t} targets x {m} anchors")

    _log("[geometry] anchor stds from kernel F ...")
    F = _load_tensor(f_path).to(torch.float64)                       # (M, N)
    if F.shape[0] != m:
        raise ValueError(f"F first dim {F.shape[0]} != embedding dim {m}")
    p_psi = F.mean(dim=1)                                            # (M,)
    std_psi = torch.sqrt((p_psi * (1.0 - p_psi)).clamp(min=0.0))     # (M,)
    del F

    _log("[geometry] streaming base rates over satisfactions ...")
    p = _base_rate_streamed(sat_path, args.chunk_size)              # (N_t,)
    variance = p * (1.0 - p)
    std_phi = np.sqrt(np.clip(variance, 0.0, None))

    # alignment = || emb / std_psi ||_2 / std_phi  (norm of the Pearson embedding).
    std_psi_safe = std_psi.clamp(min=1e-12)
    rho_scaled = (E / std_psi_safe).norm(dim=1).numpy()             # || emb_i / std_psi_i ||
    is_trivial = std_phi <= 1e-12
    std_phi_safe = np.where(is_trivial, np.nan, std_phi)
    alignment = rho_scaled / std_phi_safe
    alignment_proxy = emb_norm / std_phi_safe
    alignment = np.nan_to_num(alignment, nan=0.0)
    alignment_proxy = np.nan_to_num(alignment_proxy, nan=0.0)

    df = pd.DataFrame({
        "formula_id": np.arange(n_t, dtype=int),
        "p": p,
        "variance": variance,
        "std": std_phi,
        "emb_norm": emb_norm,
        "alignment": alignment,
        "alignment_proxy": alignment_proxy,
        "is_trivial": is_trivial.astype(int),
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    _log(f"[geometry] wrote {len(df)} rows -> {args.output}")

    # Emit the canonical trivial_ids.csv next to the dataset, so the analysis
    # loaders auto-discover and drop tautology/contradiction targets everywhere.
    triv_path = os.path.join(args.validation_dataset_dir, "trivial_ids.csv")
    df.loc[df["is_trivial"] == 1, ["formula_id"]].to_csv(triv_path, index=False)
    _log(f"[geometry] trivial (std==0) targets: {int(is_trivial.sum())} -> {triv_path}")
    _log("[geometry] feature summary:\n" + df.drop(columns=['formula_id']).describe().to_string())


if __name__ == "__main__":
    main()
