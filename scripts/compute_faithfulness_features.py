"""Compute per-target representation-faithfulness features for the validation set (study I1).

Tests whether the anchor-induced kernel preserves the TRUE covariance kernel -- i.e. whether
the embedding is a faithful realisation of the covariance kernel it is supposed to be (RQ1a,
Experiment Design ``sec:rep_faithfulness``). Judged in COVARIANCE, never Hamming.

For each validation target phi, against a fixed set of L LANDMARK partner targets psi_k sampled
from the non-trivial validation set:

  true kernel        K(phi, psi_k)  = cov(satvec_phi, satvec_psi_k)        (from full satvecs)
  anchor-induced     K~(phi, psi_k) = <emb(phi), emb(psi_k)>              (from embeddings.pt)

  relational_faithfulness = Spearman( {K~(phi,psi_k)}_k , {K(phi,psi_k)}_k )
        -- the DIRECTION half of the magnitude/direction split: "is phi's web of relationships
        preserved?" (the self pair, when phi is itself a landmark, is excluded).
  emb_norm = ||emb(phi)||,  variance = p(1-p)
        -- the MAGNITUDE / self-faithfulness half (||emb||^2 vs var); also in geometry_features.csv,
        emitted here too so the module is self-contained.

Heavy: streams the ~10 GB satisfactions tensor (mmap + chunked matmul). Caches a small CSV.

Inputs (defaults match the Snellius layout):
  --validation-dataset-dir : holds embeddings.pt (N_t x M) and satisfactions.pt (N_t x N)
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
    p.add_argument("--output", required=True, help="Output faithfulness_features.csv path")
    p.add_argument("--n-landmarks", type=int, default=256,
                   help="Number of landmark partner targets psi_k")
    p.add_argument("--chunk-size", type=int, default=500,
                   help="Rows per chunk when streaming the satisfactions tensor")
    p.add_argument("--min-variance", type=float, default=1e-9,
                   help="Landmarks are sampled only from targets with variance above this")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_tensor(path: str, *, mmap: bool = False) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", mmap=mmap) if mmap else \
          torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("embeddings", "satisfactions", "tensor", "data", "F"):
            if k in obj:
                return obj[k]
        raise ValueError(f"{path}: loaded a dict without a recognised tensor key ({list(obj)[:5]})")
    return obj


def _rank_rows(a: np.ndarray) -> np.ndarray:
    """Ordinal ranks along axis 1 (ties broken arbitrarily; covariance values rarely tie)."""
    order = a.argsort(axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(a.shape[0])[:, None]
    ranks[rows, order] = np.arange(a.shape[1])[None, :].astype(np.float64)
    return ranks


def _spearman_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-row Spearman correlation between two (C, L) matrices."""
    ra, rb = _rank_rows(A), _rank_rows(B)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    return num / np.where(den == 0.0, np.nan, den)


def _base_rate_streamed(sat: torch.Tensor, chunk: int) -> np.ndarray:
    n_t = sat.shape[0]
    p = np.empty(n_t, dtype=np.float64)
    for i0 in range(0, n_t, chunk):
        i1 = min(n_t, i0 + chunk)
        p[i0:i1] = sat[i0:i1].to(torch.float32).mean(dim=1).numpy()
    return p


def main() -> None:
    args = parse_args()
    emb_path = os.path.join(args.validation_dataset_dir, "embeddings.pt")
    sat_path = os.path.join(args.validation_dataset_dir, "satisfactions.pt")
    for pth in (emb_path, sat_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(pth)

    _log("[faith] loading embeddings ...")
    E = _load_tensor(emb_path).to(torch.float64).numpy()              # (N_t, M)
    n_t, m = E.shape
    emb_norm = np.linalg.norm(E, axis=1)                              # (N_t,)

    try:
        sat = _load_tensor(sat_path, mmap=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        _log(f"[faith] mmap load failed ({exc!r}); falling back to full load")
        sat = _load_tensor(sat_path, mmap=False)
    if sat.shape[0] != n_t:
        raise ValueError(f"satisfactions rows {sat.shape[0]} != embedding rows {n_t}")
    N = sat.shape[1]
    _log(f"[faith] {n_t} targets x {m} anchors; {N} traces")

    _log("[faith] base rates ...")
    p = _base_rate_streamed(sat, args.chunk_size)                    # (N_t,)
    variance = p * (1.0 - p)

    # --- choose landmark partner targets (non-trivial, seeded) ---
    eligible = np.where(variance > args.min_variance)[0]
    if len(eligible) == 0:
        raise ValueError("no targets with variance above --min-variance for landmarks")
    L = int(min(args.n_landmarks, len(eligible)))
    rng = np.random.default_rng(args.seed)
    land_idx = np.sort(rng.choice(eligible, size=L, replace=False))
    land_col = {int(g): j for j, g in enumerate(land_idx)}
    _log(f"[faith] {L} landmark partners")

    S_L = sat[torch.as_tensor(land_idx)].to(torch.float32).numpy()    # (L, N)
    p_L = p[land_idx]                                                 # (L,)
    E_L = E[land_idx]                                                 # (L, M)

    faith = np.full(n_t, np.nan, dtype=np.float64)
    for i0 in range(0, n_t, args.chunk_size):
        i1 = min(n_t, i0 + args.chunk_size)
        Sc = sat[i0:i1].to(torch.float32).numpy()                    # (C, N)
        joint = (Sc @ S_L.T) / float(N)                              # (C, L)  mean(chi_phi*chi_psi)
        k_true = joint - np.outer(p[i0:i1], p_L)                     # (C, L)  covariance
        k_tilde = E[i0:i1] @ E_L.T                                   # (C, L)  <emb,emb>
        f = _spearman_rows(k_tilde, k_true)
        # self-exclusion for any landmark targets in this chunk
        for local in range(i1 - i0):
            g = i0 + local
            if g in land_col:
                cols = [c for c in range(L) if c != land_col[g]]
                f[local] = _spearman_rows(k_tilde[local:local + 1, cols],
                                          k_true[local:local + 1, cols])[0]
        faith[i0:i1] = f
        if (i0 // args.chunk_size) % 10 == 0:
            _log(f"[faith] faithfulness {i1}/{n_t}")

    is_landmark = np.zeros(n_t, dtype=int)
    is_landmark[land_idx] = 1
    df = pd.DataFrame({
        "formula_id": np.arange(n_t, dtype=int),
        "relational_faithfulness": faith,
        "emb_norm": emb_norm,
        "variance": variance,
        "is_landmark": is_landmark,
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    _log(f"[faith] wrote {len(df)} rows -> {args.output}")
    _log("[faith] summary:\n" + df.drop(columns=["formula_id"]).describe().to_string())


if __name__ == "__main__":
    main()
