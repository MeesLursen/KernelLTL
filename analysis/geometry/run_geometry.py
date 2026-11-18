"""Geometry analysis of the kernel instance (thesis Sec. kernel_geometry): kernel -> tables.

Answers RQ1 without the decoder or the datasets. Every quantity is exact on the trace
sample (counts in exact fp32, everything after them in float64; see
``analysis.common.kernel_geometry``). Four parts, each written as tidy CSVs:

  rank        rank.csv, spectrum.csv, dependencies.csv
  leverage    leverage_summary.csv, leverage_by_process.csv, leverage_by_flips.csv,
              extreme_traces.csv, anchors.csv
  sets        random_sets.csv, random_sets_summary.csv
  reference   reference_differences.csv

The objects Experiment 2 uses as they are go to fixed/:

  spectrum.npz        eigenvalues (descending), eigenvectors, tolerance, rank, anchor counts
  leverage.npy        leverage of every trace (row order of traces.pt)
  trace_process.npy   process code per trace (index into PROCESS_NAMES)

plus checks.csv (hard failures unless --allow-check-failures) and manifest.json
(inputs, settings, thesis object -> file).

Usage (from the repository root, as a module, on a GPU node)::

    python -m analysis.geometry.run_geometry \
        --kernel-dir <artifacts>/kernel \
        --trace-origin <artifacts>/kernel_provenance/trace_origin.npy \
        --output-dir <artifacts>/analysis/geometry
"""

from __future__ import annotations

import sys

if not __package__:
    sys.exit("Run this as a module from the repository root:  python -m analysis.geometry.run_geometry ...")

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.common.formulas import satisfaction
from analysis.common.kernel_geometry import (PROCESS_NAMES, Projector, anchor_gram, ceiling,
                                             embedding_numerators, even_ceiling, exact_counts,
                                             flip_counts, gap_tolerance, integer_coefficients,
                                             load_kernel, resolve_device, set_exact_matmul,
                                             sparse_null_basis, spectrum, trace_processes)
from analysis.geometry.reference import (instances, share_iid, share_markov, share_mixture,
                                         share_until_sum)
from formula_utils import str_to_formula

QUANTILES = (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--kernel-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--trace-origin", type=Path, default=None,
                   help="trace_origin.npy from the one-off replay of the trace draw (kernel_provenance/; "
                        "script in git history at d49cac9); without it the "
                        "process of each trace comes from the flip-count rule.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0, help="Seed of the random sets.")
    p.add_argument("--n-sizes", type=int, default=25, help="Log-spaced set sizes from 1 to N/2.")
    p.add_argument("--n-reps", type=int, default=20, help="Random sets per size and population.")
    p.add_argument("--extreme-n", type=int, default=20, help="Traces listed at each end of the leverage range.")
    p.add_argument("--min-gap-orders", type=float, default=6.0,
                   help="Orders of magnitude the spectral gap must span to fix the rank.")
    p.add_argument("--delta", type=float, default=0.01, help="Confidence of the sample-size conditions.")
    trace = p.add_argument_group("Design values of the trace distribution (not stored in the kernel)")
    trace.add_argument("--low-variance-ratio", type=float, default=0.3, help="lambda")
    trace.add_argument("--low-var-switch-prob", type=float, default=0.1, help="p_flip")
    p.add_argument("--allow-check-failures", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    fixed = out / "fixed"
    fixed.mkdir(parents=True, exist_ok=True)
    set_exact_matmul()
    device = resolve_device(args.device)
    kd = load_kernel(args.kernel_dir, device)
    N, m = kd.N, kd.m
    print(f"[geometry] kernel: m={m} anchors, N={N} traces, device={device}", flush=True)

    checks: list[dict] = []

    def check(name: str, value, ok: bool) -> None:
        checks.append({"check": name, "value": value, "ok": bool(ok)})

    # ============================== rank ================================== #
    W, J, B = anchor_gram(kd.F)
    eig, U = spectrum(W)
    gap = gap_tolerance(eig, min_gap_orders=args.min_gap_orders)
    tol, r = gap["tol"], gap["r"]
    depths = [str_to_formula(a).depth() for a in kd.anchors]
    h = max(depths)
    ln_inv_delta = math.log(1.0 / args.delta)
    rank = {
        "M": m, "r": r, "N": N,
        "tolerance": tol, "tolerance_rule": gap["rule"], "numerical_floor": gap["floor"],
        "lambda_max": float(eig[0]), "lambda_min_nonzero": gap["smallest_kept"],
        "largest_dropped_abs": gap["largest_dropped_abs"], "gap_orders": gap["gap_orders"],
        "condition_number": float(eig[0]) / gap["smallest_kept"],
        "trace_W_sum_of_anchor_variances": float(np.trace(W)),
        "one_over_r": 1.0 / r, "r_over_N": r / N, "chance_retention_r_over_N_minus_1": r / (N - 1),
        "delta": args.delta,
        "epsilon_at_share_one_over_r": math.sqrt(2.0 * ln_inv_delta * r / N),
        "h_max_anchor_depth": h, "n_ap": kd.AP,
        "pi_id": (2 ** (h + 1) * math.log(4 * (kd.AP + 1)) + ln_inv_delta) / N,
    }
    pd.DataFrame([rank]).to_csv(out / "rank.csv", index=False)
    pd.DataFrame({"index": np.arange(m), "eigenvalue": eig,
                  "retained": eig > tol}).to_csv(out / "spectrum.csv", index=False)
    print(f"[geometry] rank r={r} of M={m}; tolerance {tol:.3e} "
          f"(gap {gap['gap_orders'] if gap['gap_orders'] is not None else 'n/a'} orders)", flush=True)

    null = U[:, eig <= tol]
    in_null_support = (np.linalg.norm(null, axis=1) > 1e-8) if null.shape[1] else np.zeros(m, bool)
    dep_rows = []
    if null.shape[1]:
        for g, v in enumerate(sparse_null_basis(null)):
            support = np.flatnonzero(v)
            coeffs = integer_coefficients(v[support])
            if coeffs is not None:
                rows_F = kd.F[torch.from_numpy(support).to(device)]
                combo = torch.from_numpy(coeffs.astype(np.float32)).to(device) @ rows_F
                exact = bool((combo == combo[0]).all())
                check(f"dependency_{g}_holds_exactly_on_every_trace", int(exact), exact)
                values = coeffs
            else:
                residual = float(v[support] @ W[np.ix_(support, support)] @ v[support])
                check(f"dependency_{g}_residual", residual, residual < 1e-12)
                values = v[support]
            for j, c in zip(support, values):
                dep_rows.append({"group": g, "anchor": int(j), "coefficient": c,
                                 "integer_coefficients": coeffs is not None,
                                 "formula": kd.anchors[j], "depth": depths[j], "rate": B[j] / N})
    pd.DataFrame(dep_rows, columns=["group", "anchor", "coefficient", "integer_coefficients",
                                    "formula", "depth", "rate"]).to_csv(out / "dependencies.csv",
                                                                        index=False)

    # ======================== basis, leverage, labels ====================== #
    proj = Projector(kd.F, B, eig, U, tol)
    lev = proj.leverage.cpu().numpy()
    check("basis_orthonormality_max_abs_error", proj.orthonormality_error(),
          proj.orthonormality_error() < 1e-9)
    check("basis_column_sums_max_abs", proj.centring_error(), proj.centring_error() < 1e-8)
    check("leverage_sum_minus_r", float(lev.sum() - r), abs(lev.sum() - r) < 1e-6)
    check("leverage_min", float(lev.min()), lev.min() > -1e-12)
    check("leverage_max", float(lev.max()), lev.max() < 1 + 1e-12)

    codes, label_info = trace_processes(kd.traces, args.trace_origin,
                                        low_variance_ratio=args.low_variance_ratio,
                                        switch_prob=args.low_var_switch_prob)
    print(f"[geometry] trace processes from {label_info['source']}", flush=True)

    np.savez(fixed / "spectrum.npz", eigenvalues=eig, eigenvectors=U, tolerance=tol, rank=r,
             anchor_counts=B, N=N)
    np.save(fixed / "leverage.npy", lev)
    np.save(fixed / "trace_process.npy", codes)

    # ============================ leverage division ========================= #
    mean = r / N
    rel = lev / mean
    desc = np.sort(lev)[::-1]

    def top_share(frac: float) -> float:
        return float(desc[:max(1, int(round(frac * N)))].sum() / r)

    summary = [("traces", N), ("r", r), ("mean_leverage_r_over_N", mean),
               ("min_relative", float(rel.min())), ("max_relative", float(rel.max()))]
    summary += [(f"quantile_{q:g}_relative", float(np.quantile(rel, q))) for q in QUANTILES]
    summary += [("budget_share_top_1pct", top_share(0.01)),
                ("budget_share_top_10pct", top_share(0.10)),
                ("share_below_half_mean", float(np.mean(rel < 0.5))),
                ("share_above_2_mean_hoaglin_welsch", float(np.mean(rel > 2.0))),
                ("share_above_3_mean", float(np.mean(rel > 3.0)))]
    pd.DataFrame(summary, columns=["statistic", "value"]).to_csv(out / "leverage_summary.csv", index=False)

    proc_rows = []
    for code, name in enumerate(PROCESS_NAMES):
        mask = codes == code
        if mask.any():
            proc_rows.append({"process": name, "traces": int(mask.sum()),
                              "trace_share": float(mask.mean()),
                              "leverage_sum": float(lev[mask].sum()),
                              "leverage_share_of_r": float(lev[mask].sum() / r),
                              "mean_relative": float(rel[mask].mean()),
                              "rows": ";".join(map(str, np.flatnonzero(mask))) if mask.sum() <= 4 else ""})
    pd.DataFrame(proc_rows).to_csv(out / "leverage_by_process.csv", index=False)

    flips = flip_counts(kd.traces).cpu().numpy()
    by_flips = (pd.DataFrame({"flips": flips, "leverage": lev, "relative": rel})
                .groupby("flips").agg(traces=("leverage", "size"),
                                      mean_relative=("relative", "mean"),
                                      leverage_sum=("leverage", "sum")).reset_index())
    by_flips["leverage_share_of_r"] = by_flips["leverage_sum"] / r
    by_flips.to_csv(out / "leverage_by_flips.csv", index=False)

    atoms_t0 = kd.traces[:, :, 0].sum(dim=1).cpu().numpy()
    true_entries = kd.traces.sum(dim=(1, 2)).cpu().numpy()
    order = np.argsort(lev, kind="stable")
    ext_rows = []
    for end, rows in (("highest", order[::-1][:args.extreme_n]), ("lowest", order[:args.extreme_n])):
        for rank_i, i in enumerate(rows, 1):
            ext_rows.append({"end": end, "rank": rank_i, "row": int(i), "leverage": float(lev[i]),
                             "relative": float(rel[i]), "process": PROCESS_NAMES[codes[i]],
                             "flips": int(flips[i]), "atoms_true_at_t0": int(atoms_t0[i]),
                             "true_entries": int(true_entries[i]), "constant_in_time": bool(flips[i] == 0)})
    pd.DataFrame(ext_rows).to_csv(out / "extreme_traces.csv", index=False)

    # Per anchor: the fraction the other anchors predict (VIF identity) and the spread
    # of its unit of leverage over the traces (partial leverage s_j[i] = h_j[i]^2/||h_j||^2,
    # h_j = row j of W^+ X_A = M_j Q^T, eqs. anchor_residual and leverage_update).
    wplus_diag = (proj.M_np ** 2).sum(axis=1)
    vif = np.diag(W) * wplus_diag
    frac_pred = np.where(in_null_support, 1.0, 1.0 - 1.0 / vif)
    n_eff = np.full(m, np.nan)
    n50 = np.full(m, np.nan)
    max_share = np.full(m, np.nan)
    unit_norm = np.full(m, np.nan)
    outside = np.flatnonzero(~in_null_support)
    for s in range(0, outside.size, 64):
        blk = outside[s:s + 64]
        H = proj.M[torch.from_numpy(blk).to(device)] @ proj.Q.T              # (b, N)
        sq = H * H
        tot = sq.sum(dim=1, keepdim=True)
        sh = sq / tot
        n_eff[blk] = (1.0 / (sh * sh).sum(dim=1)).cpu().numpy()
        srt = torch.sort(sh, dim=1, descending=True).values
        half = torch.full((blk.size, 1), 0.5, dtype=torch.float64, device=device)
        n50[blk] = (torch.searchsorted(torch.cumsum(srt, dim=1), half) + 1).squeeze(1).cpu().numpy()
        max_share[blk] = srt[:, 0].cpu().numpy()
        unit_norm[blk] = tot.squeeze(1).cpu().numpy()
        del H, sq, sh, srt
    if outside.size:
        err = float(np.max(np.abs(unit_norm[outside] / wplus_diag[outside] - 1.0)))
        check("partial_leverage_norm_equals_Wplus_diagonal_rel_error", err, err < 1e-8)
    pd.DataFrame({
        "anchor": np.arange(m), "formula": kd.anchors, "depth": depths, "rate": B / N,
        "variance": np.diag(W), "in_null_support": in_null_support,
        "fraction_predicted_by_others": frac_pred, "vif": np.where(in_null_support, np.inf, vif),
        "partial_leverage_n_eff": n_eff, "partial_leverage_n_eff_over_N": n_eff / N,
        "partial_leverage_n50": n50, "partial_leverage_max_share": max_share,
    }).to_csv(out / "anchors.csv", index=False)

    # =============================== sets ================================== #
    sizes = np.unique(np.round(np.geomspace(1, N // 2, args.n_sizes)).astype(np.int64))
    pops = {"all": np.arange(N),
            "markov": np.flatnonzero(codes == PROCESS_NAMES.index("markov")),
            "iid": np.flatnonzero(codes == PROCESS_NAMES.index("iid"))}
    set_rows = []
    for pi, (pop, members) in enumerate(pops.items()):
        cap = N // 2 if pop == "all" else members.size // 2
        for si, size in enumerate(sizes):
            if size > cap:
                continue
            for rep in range(args.n_reps):
                gen = torch.Generator().manual_seed(args.seed + 1_000_000 * pi + 1_000 * si + rep)
                pick = torch.randperm(members.size, generator=gen)[:int(size)].numpy()
                D = torch.from_numpy(members[pick])
                # Chance retention only for sets drawn from all traces: the indicator of a
                # uniformly random set is a permuted satisfaction vector, with E[R^2] =
                # r/(N-1) exactly. A random subset of one process is not unrelated to the
                # anchors (they capture the process), so it is no chance baseline.
                st = proj.set_stats(D, want_rowsum=(pop == "all"))
                share = size / N
                set_rows.append({
                    "population": pop, "size": int(size), "rep": rep, "share": share,
                    "L_D": st["L"], "l_D": st["l"], "kappa_D": ceiling(st["l"], share),
                    "mean_leverage": st["L"] / size,
                    "ceiling_min_1_L_D": min(1.0, st["L"]), "even_L_D": r * share,
                    "kappa_even": even_ceiling(r, share),
                    "chance_retention": (float((st["rowsum"] ** 2).sum()) / (size * (1.0 - share))
                                         if pop == "all" else float("nan")),
                })
        print(f"[geometry] random sets: {pop} done", flush=True)
    sets = pd.DataFrame(set_rows)
    sets.to_csv(out / "random_sets.csv", index=False)
    metrics = ["L_D", "l_D", "kappa_D", "mean_leverage", "ceiling_min_1_L_D", "chance_retention"]
    summ_rows = []
    for (pop, size), g in sets.groupby(["population", "size"], sort=False):
        row = {"population": pop, "size": size, "share": size / N, "reps": len(g),
               "even_L_D": r * size / N, "kappa_even": even_ceiling(r, size / N),
               "chance_retention_expected": r / (N - 1) if pop == "all" else float("nan")}
        for col in metrics:
            vals = g[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_q025"], row[f"{col}_q500"], row[f"{col}_q975"] = (
                (float(x) for x in np.quantile(vals, [0.025, 0.5, 0.975])) if len(vals)
                else (float("nan"),) * 3)
        summ_rows.append(row)
    pd.DataFrame(summ_rows).to_csv(out / "random_sets_summary.csv", index=False)
    ratio = sets.loc[sets["population"] == "all", "chance_retention"].mean() / (r / (N - 1))
    check("chance_retention_mean_over_expected_info_only", float(ratio), True)
    viol = int(((sets["l_D"] < sets["L_D"] / r - 1e-9) | (sets["l_D"] > np.minimum(1.0, sets["L_D"]) + 1e-9)).sum())
    check("random_sets_group_leverage_within_bounds_violations", viol, viol == 0)

    # ============================ reference ================================ #
    insts = instances(kd.AP)
    keys: dict[str, int] = {}
    sat_rows: list[torch.Tensor] = []

    def index_of(f) -> int:
        key = str(f)
        if key not in keys:
            keys[key] = len(sat_rows)
            sat_rows.append(satisfaction(f, kd.traces, time_index=kd.time_index))
        return keys[key]

    pairs = [(index_of(i.phi), index_of(i.other) if i.other is not None else None) for i in insts]
    S = torch.stack(sat_rows)
    A_ref, C_ref = exact_counts(S, kd.F)
    num_ref = embedding_numerators(A_ref, C_ref, B, N)
    anchor_match = [int(np.flatnonzero((A_ref[k] == C_ref[k]) & (B == C_ref[k]))[0])
                    if np.any((A_ref[k] == C_ref[k]) & (B == C_ref[k])) else -1
                    for k in range(len(sat_rows))]
    T, pf, lam = kd.T, args.low_var_switch_prob, args.low_variance_ratio
    for proc, val in (("iid", None), ("markov", pf)):
        closed = share_iid("until", None, T) if val is None else share_markov("until", None, T, pf)
        diff = abs(closed - share_until_sum(T, val))
        check(f"until_closed_form_equals_sum_{proc}", diff, diff < 1e-15)
    ref_rows = []
    for inst, (a, b) in zip(insts, pairs):
        sa = S[a]
        if b is not None:
            mask = sa ^ S[b]
            n_plus = int((sa & ~S[b]).sum())
            n_minus = int((~sa & S[b]).sum())
            d2 = (n_plus + n_minus) / N - ((n_plus - n_minus) / N) ** 2
            num = num_ref[a] - num_ref[b]
        else:
            mask = sa if inst.constant == "bottom" else ~sa
            rate = C_ref[a] / N
            d2 = rate * (1.0 - rate)
            num = num_ref[a]
        D = mask.nonzero().squeeze(1)
        size = int(D.numel())
        share = size / N
        st = proj.set_stats(D)
        kept = float(proj.kept_sq(num[None, :])[0])
        R2 = kept / d2 if d2 > 0 else float("nan")
        s_T = share_mixture(inst.kind, inst.param, T, pf, lam)
        ref_rows.append({
            "kind": inst.kind, "param": inst.param, "atoms": ",".join(f"p_{x}" for x in inst.atoms),
            "phi": str(inst.phi), "other": str(inst.other) if inst.other is not None else inst.constant,
            "phi_anchor": anchor_match[a], "other_anchor": anchor_match[b] if b is not None else -1,
            "D_size": size, "share_sample": share, "share_T": s_T,
            "share_iid": share_iid(inst.kind, inst.param, T),
            "share_markov": share_markov(inst.kind, inst.param, T, pf),
            "relative_shortfall": share / s_T - 1.0 if s_T > 0 else float("nan"),
            "epsilon_hoeffding": math.sqrt(2.0 * ln_inv_delta / (N * s_T)) if s_T > 0 else float("nan"),
            "one_over_r": 1.0 / r,
            "L_D": st["L"], "l_D": st["l"], "kappa_D": ceiling(st["l"], share) if size else float("nan"),
            "mean_leverage": st["L"] / size if size else float("nan"),
            "R2": R2, "even_reference": r * share / (1.0 - share) if share < 1 else float("nan"),
            "kappa_even": even_ceiling(r, share),
        })
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(out / "reference_differences.csv", index=False)
    viol = int((ref["R2"] > ref["kappa_D"] + 1e-9).sum())
    check("reference_retention_below_ceiling_violations", viol, viol == 0)

    # =========================== checks, manifest ========================= #
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out / "checks.csv", index=False)
    failed = checks_df.loc[~checks_df["ok"], "check"].tolist()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kernel_dir": str(args.kernel_dir),
        "trace_origin": str(args.trace_origin) if args.trace_origin else None,
        "trace_process_labels": label_info,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": {"torch": torch.__version__, "python": platform.python_version(),
                        "device": str(device),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "precision": "0/1 counts in fp32 (exact, TF32 off); everything after them float64",
        "fixed_objects": {name: sha256(fixed / name)
                          for name in ("spectrum.npz", "leverage.npy", "trace_process.npy")},
        "check_failures": failed,
        "thesis_objects": {
            "sec:kernel_geometry / Rank": ["rank.csv", "spectrum.csv", "dependencies.csv"],
            "sec:kernel_geometry / Leverage division": [
                "leverage_summary.csv", "leverage_by_process.csv", "leverage_by_flips.csv",
                "extreme_traces.csv", "anchors.csv"],
            "sec:geom_sets": ["random_sets.csv", "random_sets_summary.csv"],
            "tab:geom_reference (sec:geom_reference)": ["reference_differences.csv"],
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[geometry] wrote tables to {out}  (failed checks: {failed or 'none'})", flush=True)
    if failed and not args.allow_check_failures:
        raise SystemExit(f"{len(failed)} failed checks (see checks.csv): {failed}")


if __name__ == "__main__":
    main()
