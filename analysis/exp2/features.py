"""Experiment 2, step 1 of 2 (GPU): the representation quantities of targets, neighbours, misses.

Reads the kernel, the fixed objects of the geometry analysis (used as they are: the
tolerance, the rank and the eigendecomposition fix the basis), the validation dataset
and the greedy validation records, and writes the tables the statistics step reads:

  targets.csv     one row per validation target: syntax (depth, operator presence),
                  outcome (recovered, invalid), Var(phi), R^2_phi with the ceiling of the
                  target's minority set, and the neighbour aggregates for every k in --ks
  neighbours.csv  one row per (target, rank) over the k_max nearest validation targets
  misses.csv      one row per valid greedy output that is not equivalent to its target,
                  with the three terms of thesis eq. miss_terms

plus checks.csv and manifest.json. Every quantity comes from the satisfaction vectors in
exact counts and float64; the embeddings the decoder received are only used to check
that the counts reproduce them bit for bit.

Usage (from the repository root, as a module, on a GPU node)::

    python -m analysis.exp2.features \
        --kernel-dir <artifacts>/kernel \
        --geometry-dir <artifacts>/analysis/geometry \
        --dataset-dir <artifacts>/datasets/validation \
        --run-dir <artifacts>/validation/ce_base \
        --output-dir <artifacts>/analysis/exp2/features
"""

from __future__ import annotations

import sys

if not __package__:
    sys.exit("Run this as a module from the repository root:  python -m analysis.exp2.features ...")

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.common.formulas import OPERATOR_NAMES, operator_presence, satisfaction
from analysis.common.kernel_geometry import (Projector, ceiling, embedding_numerators,
                                             even_ceiling, exact_counts, load_kernel,
                                             pairwise_counts, resolve_device, set_exact_matmul)
from analysis.common.load import check_alignment, load_greedy
from formula_utils import str_to_formula


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--kernel-dir", required=True, type=Path)
    p.add_argument("--geometry-dir", required=True, type=Path,
                   help="Output dir of analysis.geometry.run_geometry (its fixed/ objects are used).")
    p.add_argument("--dataset-dir", required=True, type=Path, help="Validation dataset dir.")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Conditioned validation run dir (per_sample/greedy.jsonl).")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--ks", default="1,5,10",
                   help="Neighbour counts for the aggregates; the largest is the one reported.")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-check-failures", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ks = sorted({int(k) for k in args.ks.split(",")})
    k_max = ks[-1]
    set_exact_matmul()
    device = resolve_device(args.device)
    kd = load_kernel(args.kernel_dir, device)
    N, m = kd.N, kd.m
    checks: list[dict] = []

    def check(name: str, value, ok: bool) -> None:
        checks.append({"check": name, "value": value, "ok": bool(ok)})

    # ------------------------- basis from the geometry run ------------------ #
    spec = np.load(args.geometry_dir / "fixed" / "spectrum.npz")
    B = exact_counts(kd.F, kd.F)[1]
    check("anchor_counts_match_geometry", int(np.sum(B != spec["anchor_counts"])),
          np.array_equal(B, spec["anchor_counts"]))
    proj = Projector(kd.F, B, spec["eigenvalues"], spec["eigenvectors"], float(spec["tolerance"]))
    r = proj.r
    lev_diff = float(np.max(np.abs(proj.leverage.cpu().numpy()
                                   - np.load(args.geometry_dir / "fixed" / "leverage.npy"))))
    check("rank_matches_geometry", r, r == int(spec["rank"]))
    check("leverage_matches_geometry_max_abs", lev_diff, lev_diff < 1e-12)
    print(f"[exp2-features] basis rebuilt: r={r}, N={N}, device={device}", flush=True)

    # ------------------------------ validation data -------------------------- #
    meta = json.loads((args.dataset_dir / "metadata.json").read_text())
    n = int(meta["size"])
    formulas = [line.rstrip("\n") for line in open(args.dataset_dir / "formulas.jsonl") if line.strip()]
    emb = torch.load(args.dataset_dir / "embeddings.pt", map_location="cpu")
    S = torch.load(args.dataset_dir / "satisfactions.pt", map_location="cpu").to(torch.bool).to(device)
    if S.shape != (n, N) or len(formulas) != n:
        raise SystemExit(f"dataset shapes {tuple(S.shape)} / {len(formulas)} do not match n={n}, N={N}")
    greedy, load_checks = load_greedy(args.run_dir, expected_n=n)
    check_alignment(greedy, args.dataset_dir, name=f"{args.run_dir.name}/greedy")
    for c in load_checks:
        check(f"records_{c['check']}", c["violations"], c["violations"] == 0)
    greedy = greedy.set_index("formula_id").sort_index()

    # ---------------------------------- targets ------------------------------ #
    A, C = exact_counts(S, kd.F)
    num = embedding_numerators(A, C, B, N)
    rebuilt = torch.from_numpy(num.astype(np.float64) / float(N * N)).to(torch.float32)
    mism = int((rebuilt != emb).any(dim=1).sum())
    check("embeddings_reproduced_bit_for_bit_mismatched_rows", mism, mism == 0)
    rate = C / N
    var = rate * (1.0 - rate)
    kept_phi = proj.kept_sq(num)
    satisfied_side = C <= N - C
    minority = np.minimum(C, N - C) / N
    t_rows = []
    for i in range(n):
        D = (S[i] if satisfied_side[i] else ~S[i]).nonzero().squeeze(1)
        st = proj.set_stats(D)
        f = str_to_formula(formulas[i])
        t_rows.append({
            "formula_id": i, "formula": formulas[i], "depth": f.depth(),
            **{f"op_{k}": v for k, v in operator_presence(f).items()},
            "recovered": int(greedy.at[i, "is_semantic_equivalent"]),
            "invalid": int(greedy.at[i, "is_invalid"]),
            "rate": rate[i], "var": var[i], "minority_share": minority[i],
            "minority_side": "satisfying" if satisfied_side[i] else "falsifying",
            "kept_phi": kept_phi[i], "R2_phi": kept_phi[i] / var[i],
            "L_D_phi": st["L"], "l_D_phi": st["l"], "kappa_phi": ceiling(st["l"], minority[i]),
            "mean_leverage_phi": st["L"] / st["size"], "kappa_even_phi": even_ceiling(r, minority[i]),
        })
        if (i + 1) % 1000 == 0:
            print(f"[exp2-features] targets: {i + 1}/{n}", flush=True)
    targets = pd.DataFrame(t_rows)

    # -------------------------------- neighbours ----------------------------- #
    J = pairwise_counts(S)
    size = C[:, None] + C[None, :] - 2 * J
    d2 = size / N - ((C[:, None] - C[None, :]) / N) ** 2
    np.fill_diagonal(d2, np.inf)
    off = size[~np.eye(n, dtype=bool)]
    check("targets_pairwise_distinct_min_disagreement", int(off.min()), off.min() > 0)
    order = np.argsort(d2, axis=1, kind="stable")[:, :k_max]          # ties broken by formula_id
    pair_keys = sorted({(min(a, b), max(a, b)) for a in range(n) for b in order[a]})
    diffs = np.stack([num[a] - num[b] for a, b in pair_keys])
    kept_pairs = proj.kept_sq(diffs)
    pair_stats: dict[tuple[int, int], dict] = {}
    for idx, (a, b) in enumerate(pair_keys):
        st = proj.set_stats((S[a] ^ S[b]).nonzero().squeeze(1))
        share = int(size[a, b]) / N
        pair_stats[(a, b)] = {"kept": kept_pairs[idx], "L_D": st["L"], "l_D": st["l"],
                              "kappa_D": ceiling(st["l"], share), "mean_leverage": st["L"] / st["size"],
                              "kappa_even": even_ceiling(r, share)}
        if (idx + 1) % 5000 == 0:
            print(f"[exp2-features] neighbour pairs: {idx + 1}/{len(pair_keys)}", flush=True)
    n_rows = []
    for a in range(n):
        for rank_i, b in enumerate(order[a], 1):
            ps = pair_stats[(min(a, b), max(a, b))]
            share = int(size[a, b]) / N
            n_rows.append({
                "formula_id": a, "rank": rank_i, "neighbour_id": int(b), "d2": d2[a, b],
                "D_size": int(size[a, b]), "share": share,
                "n_plus": int(C[a] - J[a, b]), "n_minus": int(C[b] - J[a, b]),
                "kept": ps["kept"], "R2": ps["kept"] / d2[a, b], "L_D": ps["L_D"], "l_D": ps["l_D"],
                "kappa_D": ps["kappa_D"], "mean_leverage": ps["mean_leverage"],
                "kappa_even": ps["kappa_even"],
            })
    neighbours = pd.DataFrame(n_rows)
    neighbours["above_chance"] = (neighbours["R2"] > neighbours["mean_leverage"]).astype(float)
    weighted_kappa = neighbours["kappa_D"] * neighbours["d2"]
    for k in ks:
        sel = neighbours["rank"] <= k
        g = neighbours[sel].groupby("formula_id")
        d2_sum = g["d2"].sum()
        agg = pd.DataFrame({
            f"mean_d2_k{k}": g["d2"].mean(),
            # pooled = ratio of sums: the fraction of the summed squared distance that is kept
            f"pooled_R2_k{k}": g["kept"].sum() / d2_sum,
            # the pooled retention if every pair kept up to its own ceiling
            f"pooled_kappa_k{k}": weighted_kappa[sel].groupby(neighbours.loc[sel, "formula_id"]).sum() / d2_sum,
            f"mean_share_k{k}": g["share"].mean(),
            f"frac_above_chance_k{k}": g["above_chance"].mean(),
        })
        targets = targets.merge(agg, left_on="formula_id", right_index=True, how="left")

    # ---------------------------------- misses ------------------------------- #
    miss_rec = greedy[(~greedy["is_invalid"]) & (~greedy["is_semantic_equivalent"])]
    fids = np.array(miss_rec.index.to_numpy(), dtype=np.int64)
    So = torch.stack([satisfaction(str_to_formula(s), kd.traces, time_index=kd.time_index)
                      for s in miss_rec["generated_formula_str"]])
    Ao, Co = exact_counts(So, kd.F)
    num_o = embedding_numerators(Ao, Co, B, N)
    J_to = (S[torch.from_numpy(fids).to(device)] & So).sum(dim=1).cpu().numpy().astype(np.int64)
    n_plus = C[fids] - J_to
    n_minus = Co - J_to
    m_size = n_plus + n_minus
    recorded = miss_rec["semantic_distance"].to_numpy()
    bad = int(np.sum(np.abs(recorded * N - m_size) >= 0.5))
    check("miss_recorded_distance_reproduced_mismatches", bad, bad == 0)
    check("misses_have_nonempty_disagreement", int((m_size == 0).sum()), bool((m_size > 0).all()))
    m_d2 = m_size / N - ((n_plus - n_minus) / N) ** 2
    m_kept = proj.kept_sq(num[fids] - num_o)
    mi_rows = []
    for j, fid in enumerate(fids):
        st = proj.set_stats((S[int(fid)] ^ So[j]).nonzero().squeeze(1))
        share = int(m_size[j]) / N
        R2 = m_kept[j] / m_d2[j]
        kap = ceiling(st["l"], share)
        mi_rows.append({
            "formula_id": int(fid), "depth": int(miss_rec.at[fid, "target_depth"]),
            "generated_formula": miss_rec.at[fid, "generated_formula_str"],
            "generated_depth": int(miss_rec.at[fid, "generated_depth"]),
            "D_size": int(m_size[j]), "share": share, "n_plus": int(n_plus[j]), "n_minus": int(n_minus[j]),
            "d2": m_d2[j], "kept": m_kept[j], "R2": R2, "L_D": st["L"], "l_D": st["l"],
            "kappa_D": kap, "mean_leverage": st["L"] / st["size"], "even_L_D": r * share,
            "kappa_even": even_ceiling(r, share),
            "out_of_reach": 1.0 - kap, "within_reach_not_kept": kap - R2, "kept_term": R2,
            "largest_shift": kap - st["l"],
        })
        if (j + 1) % 500 == 0:
            print(f"[exp2-features] misses: {j + 1}/{len(fids)}", flush=True)
    misses = pd.DataFrame(mi_rows)

    # --------------------------------- checks -------------------------------- #
    tol = 1e-9
    for name, frame, r2, kap in (("targets", targets, "R2_phi", "kappa_phi"),
                                 ("neighbours", neighbours, "R2", "kappa_D"),
                                 ("misses", misses, "R2", "kappa_D")):
        v = int(((frame[r2] < -tol) | (frame[r2] > frame[kap] + tol)).sum())
        check(f"{name}_retention_within_0_and_ceiling_violations", v, v == 0)
    for name, frame, L, l in (("targets", targets, "L_D_phi", "l_D_phi"),
                              ("neighbours", neighbours, "L_D", "l_D"),
                              ("misses", misses, "L_D", "l_D")):
        v = int(((frame[l] < frame[L] / r - tol) | (frame[l] > np.minimum(1.0, frame[L]) + tol)).sum())
        check(f"{name}_group_leverage_within_bounds_violations", v, v == 0)

    targets.to_csv(out / "targets.csv", index=False)
    neighbours.to_csv(out / "neighbours.csv", index=False)
    misses.to_csv(out / "misses.csv", index=False)
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out / "checks.csv", index=False)
    failed = checks_df.loc[~checks_df["ok"], "check"].tolist()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"kernel_dir": str(args.kernel_dir), "geometry_dir": str(args.geometry_dir),
                   "dataset_dir": str(args.dataset_dir), "run_dir": str(args.run_dir)},
        "r": r, "N": N, "n_targets": n, "ks": ks,
        "counts": {"recovered": int(targets["recovered"].sum()), "invalid": int(targets["invalid"].sum()),
                   "misses": int(len(misses)), "neighbour_pairs": len(pair_keys)},
        "operators": list(OPERATOR_NAMES),
        "environment": {"torch": torch.__version__, "python": platform.python_version(),
                        "device": str(device),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "precision": "0/1 counts in fp32 (exact, TF32 off); everything after them float64",
        "check_failures": failed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[exp2-features] wrote {out}  (failed checks: {failed or 'none'})", flush=True)
    if failed and not args.allow_check_failures:
        raise SystemExit(f"{len(failed)} failed checks (see checks.csv): {failed}")


if __name__ == "__main__":
    main()
