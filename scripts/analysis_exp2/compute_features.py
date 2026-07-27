"""Per-target feature extraction for Experiment 2 (geometry + faithfulness).

The HEAVY step of Part II: touches the large satisfactions tensor once and
caches small artifacts that the light analysis step (run_exp2.py) consumes.

Per validation target phi (keyed by ``formula_id`` = row index):

  p             base rate mean(satvec)
  variance      p(1-p)  (Bernoulli variance; informativeness driver)
  std           sqrt(variance)
  emb_norm      ||emb(phi)||_2  (raw conditioning magnitude)
  alignment_proxy  emb_norm / std_phi  (parametric variance-normalised norm;
                robustness companion to the binned residual built downstream)
  relational_faithfulness
                Spearman( <emb(phi), emb(psi_k)> , cov(satvec_phi, satvec_psi_k) )
                over L fixed landmark partner targets psi_k (self pair excluded)
  is_landmark   1 iff phi is one of the landmark partners

Also dumped, so that alternative faithfulness correlations (e.g. Pearson
instead of Spearman) stay post-hoc without re-touching the satisfactions
tensor:

  k_true.npy         (N_t, L)  true covariances cov(phi, psi_k)
  k_tilde.npy        (N_t, L)  embedding inner products <emb(phi), emb(psi_k)>
  landmark_ids.npy   (L,)      formula_ids of the landmark partners

The residual-norm covariate u is deliberately NOT cached here: it is
re-estimated inside every bootstrap resample by the analysis step, so the
intervals absorb the binning uncertainty (generated-regressor fix).

Built-in consistency gate (--verify-sample): recomputes emb(phi) = F_c @ satvec_c / N
for a random sample of targets and compares against the stored embeddings. A
mismatch means the kernel directory does not belong to the dataset (wrong trace
sample) and the run aborts before any heavy work: every downstream number
would be meaningless.

Trivial targets are excluded from the dataset at build time; this script
asserts that (variance > 0 for every row) rather than emitting trivial flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--validation-dataset-dir", required=True, type=Path,
                   help="Dir with embeddings.pt and satisfactions.pt (validation LTLDataset)")
    p.add_argument("--kernel-dir", required=True, type=Path,
                   help="Dir with F.pt (anchor x trace satisfaction matrix)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-landmarks", type=int, default=256)
    p.add_argument("--row-chunk", type=int, default=500,
                   help="Target rows per chunk when streaming satisfactions")
    p.add_argument("--trace-chunk", type=int, default=100_000,
                   help="Trace columns per chunk for the verification gate")
    p.add_argument("--verify-sample", type=int, default=16,
                   help="Targets sampled for the embedding-recomputation gate (0 disables)")
    p.add_argument("--verify-rtol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_tensor(path: Path, *, mmap: bool = False) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", mmap=mmap)
    if isinstance(obj, dict):
        for k in ("embeddings", "satisfactions", "tensor", "data", "F"):
            if k in obj:
                return obj[k]
        raise ValueError(f"{path}: dict without a recognised tensor key ({list(obj)[:5]})")
    return obj


def _rank_rows(a: np.ndarray) -> np.ndarray:
    """Ordinal ranks along axis 1 (covariance values essentially never tie)."""
    order = a.argsort(axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(a.shape[0])[:, None]
    ranks[rows, order] = np.arange(a.shape[1])[None, :].astype(np.float64)
    return ranks


def _spearman_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ra, rb = _rank_rows(A), _rank_rows(B)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    return num / np.where(den == 0.0, np.nan, den)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    emb_path = args.validation_dataset_dir / "embeddings.pt"
    sat_path = args.validation_dataset_dir / "satisfactions.pt"
    f_path = args.kernel_dir / "F.pt"
    for pth in (emb_path, sat_path, f_path):
        if not pth.exists():
            raise FileNotFoundError(pth)

    _log("[exp2] loading embeddings ...")
    E = _load_tensor(emb_path).to(torch.float64).numpy()             # (N_t, M)
    n_t, m = E.shape
    emb_norm = np.linalg.norm(E, axis=1)

    try:
        sat = _load_tensor(sat_path, mmap=True)
    except Exception as exc:
        _log(f"[exp2] mmap load failed ({exc!r}); falling back to full load")
        sat = _load_tensor(sat_path, mmap=False)
    if sat.shape[0] != n_t:
        raise ValueError(f"satisfactions rows {sat.shape[0]} != embedding rows {n_t}")
    N = sat.shape[1]
    _log(f"[exp2] {n_t} targets x {m} anchors; {N} traces")

    F = _load_tensor(f_path, mmap=True)                              # (M, N) bool
    if F.shape[0] != m:
        raise ValueError(f"F first dim {F.shape[0]} != embedding dim {m}")
    if F.shape[1] != N:
        raise ValueError(f"F trace dim {F.shape[1]} != satisfactions trace dim {N}")

    # ---- embedding-recomputation gate (before any heavy streaming) --------- #
    worst = None
    if args.verify_sample > 0:
        rng = np.random.default_rng(args.seed)
        sample = np.sort(rng.choice(n_t, size=min(args.verify_sample, n_t), replace=False))
        _log(f"[exp2] verifying stored embeddings on {len(sample)} targets ...")
        worst = 0.0
        for g in sample:
            chi = sat[int(g)].to(torch.float64).numpy()              # (N,)
            chi_c = chi - chi.mean()
            recomputed = np.zeros(m, dtype=np.float64)
            for t0 in range(0, N, args.trace_chunk):
                t1 = min(N, t0 + args.trace_chunk)
                recomputed += F[:, t0:t1].to(torch.float64).numpy() @ chi_c[t0:t1]
            recomputed /= N                                          # = F_c @ chi_c / N
            denom = max(float(np.linalg.norm(E[g])), 1e-12)
            rel = float(np.linalg.norm(recomputed - E[g]) / denom)
            worst = max(worst, rel)
        _log(f"[exp2] worst relative embedding error: {worst:.3e}")
        if worst > args.verify_rtol:
            raise ValueError(
                f"stored embeddings disagree with F_c @ satvec_c / N "
                f"(worst rel err {worst:.3e} > {args.verify_rtol}); the kernel dir "
                "does not match the dataset's trace sample -- aborting.")

    _log("[exp2] streaming base rates ...")
    p = np.empty(n_t, dtype=np.float64)
    for i0 in range(0, n_t, args.row_chunk):
        i1 = min(n_t, i0 + args.row_chunk)
        p[i0:i1] = sat[i0:i1].to(torch.float32).mean(dim=1).numpy()
    variance = p * (1.0 - p)
    std_phi = np.sqrt(np.clip(variance, 0.0, None))
    if (variance <= 0.0).any():
        bad = np.where(variance <= 0.0)[0]
        raise ValueError(
            f"{len(bad)} targets have constant satvecs (e.g. ids {bad[:5].tolist()}) -- "
            "trivial targets should have been excluded at dataset build time.")

    alignment_proxy = emb_norm / std_phi

    # ---- landmarks + faithfulness ----------------------------------------- #
    rng = np.random.default_rng(args.seed)
    L = int(min(args.n_landmarks, n_t))
    land_idx = np.sort(rng.choice(n_t, size=L, replace=False))
    land_col = {int(g): j for j, g in enumerate(land_idx)}
    _log(f"[exp2] {L} landmark partners")

    S_L = sat[torch.as_tensor(land_idx)].to(torch.float32).numpy()   # (L, N)
    p_L = p[land_idx]
    E_L = E[land_idx]

    k_true = np.empty((n_t, L), dtype=np.float64)
    k_tilde = np.empty((n_t, L), dtype=np.float64)
    faith = np.full(n_t, np.nan, dtype=np.float64)
    for i0 in range(0, n_t, args.row_chunk):
        i1 = min(n_t, i0 + args.row_chunk)
        Sc = sat[i0:i1].to(torch.float32).numpy()                    # (C, N)
        joint = (Sc @ S_L.T) / float(N)
        k_true[i0:i1] = joint - np.outer(p[i0:i1], p_L)
        k_tilde[i0:i1] = E[i0:i1] @ E_L.T
        f = _spearman_rows(k_tilde[i0:i1], k_true[i0:i1])
        for local in range(i1 - i0):                                  # self-exclusion
            g = i0 + local
            if g in land_col:
                cols = [c for c in range(L) if c != land_col[g]]
                f[local] = _spearman_rows(k_tilde[g:g + 1, cols],
                                          k_true[g:g + 1, cols])[0]
        faith[i0:i1] = f
        _log(f"[exp2] faithfulness {i1}/{n_t}")

    np.save(out / "k_true.npy", k_true)
    np.save(out / "k_tilde.npy", k_tilde)
    np.save(out / "landmark_ids.npy", land_idx)

    is_landmark = np.zeros(n_t, dtype=int)
    is_landmark[land_idx] = 1
    df = pd.DataFrame({
        "formula_id": np.arange(n_t, dtype=int),
        "p": p, "variance": variance, "std": std_phi,
        "emb_norm": emb_norm,
        "alignment_proxy": alignment_proxy,
        "relational_faithfulness": faith,
        "is_landmark": is_landmark,
    })
    df.to_csv(out / "exp2_features.csv", index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_dataset_dir": str(args.validation_dataset_dir),
        "kernel_dir": str(args.kernel_dir),
        "n_targets": n_t, "n_anchors": m, "n_traces": N,
        "n_landmarks": L, "seed": args.seed,
        "embedding_verification": {
            "sample": args.verify_sample, "rtol": args.verify_rtol,
            "worst_rel_err": worst,
            "passed": (worst is not None and worst <= args.verify_rtol),
        },
        "files": ["exp2_features.csv", "k_true.npy", "k_tilde.npy",
                  "landmark_ids.npy"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(f"[exp2] wrote features for {n_t} targets -> {out}")
    _log("[exp2] summary:\n" + df.drop(columns=["formula_id"]).describe().to_string())


if __name__ == "__main__":
    main()
