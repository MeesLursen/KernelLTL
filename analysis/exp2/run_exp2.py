"""Experiment 2, step 2 of 2 (CPU): feature tables -> statistical tables.

Reads the tables written by ``analysis.exp2.features`` and emits tidy CSVs with point
estimates and 95% percentile-bootstrap intervals over targets (thesis Sec. "Statistical
conventions"; slice seeds as in Experiment 1):

  keeps_by_slice.csv      what the representation keeps of the targets, overall, by
                          depth and by operator (per-target means)
  keeps_counts.csv        counts quoted in the text (e.g. targets with Var < 1/r)
  miss_terms.csv          tab:miss_terms: mean of each term of eq. miss_terms + kappa_D - l_D
  miss_shape.csv          quantiles and histogram of the kept fraction R^2 over the misses
  miss_chance.csv         R^2 against the chance level L_D/|D|
  miss_even.csv           L_D against its even-division value r|D|/N, and the counts
                          L_D < 1 vs |D|/N < 1/r
  miss_size.csv           the three terms by disagreement share
  regression_*.csv        the regression ladder (see analysis.exp2.regression)
  checks.csv, manifest.json

Usage (from the repository root, as a module)::

    python -m analysis.exp2.run_exp2 \
        --features-dir <artifacts>/analysis/exp2/features \
        --output-dir <artifacts>/analysis/exp2 --workers 16
"""

from __future__ import annotations

import sys

if not __package__:
    sys.exit("Run this as a module from the repository root:  python -m analysis.exp2.run_exp2 ...")

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis.common.bootstrap import DEFAULT_B, DEFAULT_SEED, index_matrix, mean_ci
from analysis.common.formulas import OPERATOR_NAMES
from analysis.exp2 import regression

SHAPE_Q = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--features-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--k", type=int, default=10, help="Neighbour count of the reported covariates.")
    p.add_argument("--sensitivity-ks", default="1,5")
    p.add_argument("--bootstrap-samples", type=int, default=DEFAULT_B)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--workers", type=int, default=1, help="Processes for the regression bootstrap.")
    p.add_argument("--allow-check-failures", action="store_true")
    return p.parse_args()


def slice_rows(values: dict[str, np.ndarray], mask: np.ndarray, *, seed: int, b: int,
               label: dict) -> list[dict]:
    """Estimate + CI rows for one slice; one index matrix shared by all metrics of the slice."""
    idx = index_matrix(int(mask.sum()), b=b, seed=seed)
    rows = []
    for metric, v in values.items():
        est, lo, hi, n_eff = mean_ci(np.asarray(v, dtype=np.float64)[mask], idx)
        rows.append({**label, "metric": metric, "estimate": est, "ci_lo": lo, "ci_hi": hi,
                     "n": int(mask.sum()), "n_effective": n_eff})
    return rows


def depth_slices(depth: np.ndarray, seed: int):
    yield {"slice": "all", "target_depth": "all"}, np.ones(depth.size, bool), seed
    for d in sorted(np.unique(depth)):
        yield {"slice": "depth", "target_depth": int(d)}, depth == d, seed + 100 * int(d)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fdir = args.features_dir
    fman = json.loads((fdir / "manifest.json").read_text())
    r, N = int(fman["r"]), int(fman["N"])
    K = args.k
    targets = pd.read_csv(fdir / "targets.csv")
    misses = pd.read_csv(fdir / "misses.csv")
    B = args.bootstrap_samples
    checks: list[dict] = []

    def check(name: str, value, ok: bool) -> None:
        checks.append({"check": name, "value": value, "ok": bool(ok)})

    # ------------------------------ keeps ---------------------------------- #
    t = targets
    gap_phi = t["kappa_phi"] - t["R2_phi"]
    keep_vals = {
        "var": t["var"], "R2_phi": t["R2_phi"], "kappa_phi": t["kappa_phi"],
        "gap_to_ceiling_phi": gap_phi, "at_ceiling_phi": (gap_phi <= 1e-9).astype(float),
        "minority_share_below_1_over_r_plus_1": (t["minority_share"] < 1.0 / (r + 1)).astype(float),
        "var_below_1_over_r": (t["var"] < 1.0 / r).astype(float),
        f"mean_d2_k{K}": t[f"mean_d2_k{K}"], f"mean_share_k{K}": t[f"mean_share_k{K}"],
        f"pooled_R2_k{K}": t[f"pooled_R2_k{K}"], f"pooled_kappa_k{K}": t[f"pooled_kappa_k{K}"],
        f"frac_neighbours_above_chance_k{K}": t[f"frac_above_chance_k{K}"],
        "recovered": t["recovered"],
    }
    keep_vals = {k: np.asarray(v, dtype=np.float64) for k, v in keep_vals.items()}
    depth = t["depth"].to_numpy()
    rows = []
    for label, mask, seed in depth_slices(depth, args.seed):
        rows += slice_rows(keep_vals, mask, seed=seed, b=B, label={**label, "operator": "any"})
    for j, o in enumerate(OPERATOR_NAMES):
        mask = t[f"op_{o}"].to_numpy() == 1
        rows += slice_rows(keep_vals, mask, seed=args.seed + 1000 + j, b=B,
                           label={"slice": "operator", "target_depth": "all", "operator": o})
    pd.DataFrame(rows).to_csv(out / "keeps_by_slice.csv", index=False)
    counts = {"targets": len(t), "recovered": int(t["recovered"].sum()), "invalid": int(t["invalid"].sum()),
              "misses": len(misses), "var_below_1_over_r": int((t["var"] < 1.0 / r).sum()),
              "minority_share_below_1_over_r_plus_1": int((t["minority_share"] < 1.0 / (r + 1)).sum()),
              "R2_phi_at_ceiling": int((gap_phi <= 1e-9).sum()), "r": r, "N": N}
    pd.DataFrame([counts]).to_csv(out / "keeps_counts.csv", index=False)

    # ------------------------------ misses --------------------------------- #
    mi = misses
    total = mi["out_of_reach"] + mi["within_reach_not_kept"] + mi["kept_term"]
    err = float(np.max(np.abs(total - 1.0))) if len(mi) else 0.0
    check("miss_terms_sum_to_one_max_abs_error", err, err < 1e-12)
    shift_ok = bool(((mi["largest_shift"] >= -1e-12) & (mi["largest_shift"] <= mi["share"] + 1e-12)).all())
    check("largest_shift_within_0_and_share", shift_ok, shift_ok)
    mdepth = mi["depth"].to_numpy()
    term_vals = {k: mi[k].to_numpy(np.float64) for k in
                 ("out_of_reach", "within_reach_not_kept", "kept_term", "largest_shift")}
    rows, shape_rows, chance_rows, even_rows = [], [], [], []
    above = (mi["R2"] > mi["mean_leverage"]).to_numpy(np.float64)
    below_even = (mi["L_D"] < mi["even_L_D"]).to_numpy(np.float64)
    for label, mask, seed in depth_slices(mdepth, args.seed):
        rows += slice_rows(term_vals, mask, seed=seed, b=B, label=label)
        kept = mi.loc[mask, "kept_term"].to_numpy()
        shape_rows += [{**label, "kind": "quantile", "at": q, "value": float(np.quantile(kept, q))}
                       for q in SHAPE_Q]
        hist, edges = np.histogram(kept, bins=20, range=(0.0, 1.0))
        shape_rows += [{**label, "kind": "histogram", "at": float(edges[i]), "value": int(hist[i])}
                       for i in range(20)]
        chance_rows += slice_rows({"above_chance": above}, mask, seed=seed, b=B, label=label)
        chance_rows.append({**label, "metric": "median_R2_over_mean_leverage",
                            "estimate": float(np.median(mi.loc[mask, "R2"] / mi.loc[mask, "mean_leverage"])),
                            "n": int(mask.sum())})
        even_rows += slice_rows({"L_D_below_even": below_even}, mask, seed=seed, b=B, label=label)
        sub = mi[mask]
        even_rows += [{**label, "metric": m, "estimate": v, "n": int(mask.sum())} for m, v in (
            ("median_L_D_over_even", float(np.median(sub["L_D"] / sub["even_L_D"]))),
            ("count_L_D_below_1", int((sub["L_D"] < 1).sum())),
            ("count_share_below_1_over_r", int((sub["share"] < 1.0 / r).sum())),
            ("count_both", int(((sub["L_D"] < 1) & (sub["share"] < 1.0 / r)).sum())))]
    pd.DataFrame(rows).to_csv(out / "miss_terms.csv", index=False)
    pd.DataFrame(shape_rows).to_csv(out / "miss_shape.csv", index=False)
    pd.DataFrame(chance_rows).to_csv(out / "miss_chance.csv", index=False)
    pd.DataFrame(even_rows).to_csv(out / "miss_even.csv", index=False)

    size_edges = [0.0, *np.unique([1.0 / r, 10.0 / r, 0.01, 0.1]), 1.0 + 1e-12]
    size_bin = np.digitize(mi["share"].to_numpy(), size_edges[1:-1])
    rows = []
    for bi in range(len(size_edges) - 1):
        mask = size_bin == bi
        if mask.any():
            rows += slice_rows(term_vals, mask, seed=args.seed + 2000 + bi, b=B,
                               label={"share_from": size_edges[bi], "share_to": min(size_edges[bi + 1], 1.0)})
    rho = stats.spearmanr(mi["share"], mi["out_of_reach"]) if len(mi) > 2 else None
    rows.append({"share_from": 0.0, "share_to": 1.0, "metric": "spearman_share_vs_out_of_reach",
                 "estimate": float(rho.statistic) if rho is not None else np.nan,
                 "ci_lo": np.nan, "ci_hi": np.nan, "n": len(mi), "n_effective": len(mi),
                 "p_value": float(rho.pvalue) if rho is not None else np.nan})
    pd.DataFrame(rows).to_csv(out / "miss_size.csv", index=False)

    # ----------------------------- regression ------------------------------ #
    sens = [int(k) for k in args.sensitivity_ks.split(",") if k.strip()]
    res = regression.run(targets, k=K, sensitivity_ks=sens, b=B, seed=args.seed, workers=args.workers)
    for name, frame in res.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(out / f"regression_{name}.csv", index=False)
    flagged = int(res["separation"]["flagged_resamples"].sum())
    check("regression_flagged_resamples_info_only", flagged, True)

    # --------------------------- checks, manifest -------------------------- #
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out / "checks.csv", index=False)
    failed = checks_df.loc[~checks_df["ok"], "check"].tolist()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features_dir": str(fdir), "features_manifest": fman,
        "k": K, "sensitivity_ks": sens, "regression_forms": res["forms"],
        "bootstrap": {"samples": B, "seed": args.seed, "interval": "percentile 2.5/97.5",
                      "slice_seed_rule": "all: seed; depth d: seed + 100 d; operator j: seed + 1000 + j; "
                                         "share bin b: seed + 2000 + b; regression: seed over all targets",
                      "held_fixed": "every target's quantities, the transforms and the decile edges"},
        "check_failures": failed,
        "thesis_objects": {
            "Exp 2 / What the representation keeps": ["keeps_by_slice.csv", "keeps_counts.csv",
                                                       "features/targets.csv", "features/neighbours.csv"],
            "fig:miss_scatter": {"file": "features/misses.csv",
                                 "x": "kept_term", "y": "kappa_D", "shade": "share"},
            "tab:miss_terms": ["miss_terms.csv"],
            "Exp 2 / misses: shape, chance, even division, size": [
                "miss_shape.csv", "miss_chance.csv", "miss_even.csv", "miss_size.csv"],
            "eq:diag_ladder, tab:diag_readings": [
                "regression_coefficients.csv", "regression_readings.csv", "regression_block_test.csv",
                "regression_predictions.csv"],
            "Exp 2 / regression checks": [
                "regression_spec_search.csv", "regression_adequacy.csv", "regression_occupancy_summary.csv",
                "regression_occupancy_cells.csv", "regression_positivity.csv", "regression_separation.csv",
                "regression_design_sparsity.csv", "regression_sensitivity.csv"],
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[exp2] wrote tables to {out}  (failed checks: {failed or 'none'})", flush=True)
    if failed and not args.allow_check_failures:
        raise SystemExit(f"{len(failed)} failed checks (see checks.csv): {failed}")


if __name__ == "__main__":
    main()
