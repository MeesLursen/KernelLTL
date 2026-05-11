"""Standalone dataset analysis for any LTLDataset directory.

Outputs:
  <output-dir>/
    figures/*.{pdf,png}     # ~16 figures + shape-uniformity figures
    dataset_summary.csv     # per-depth descriptives table
    shape_uniformity.json   # entropy ratios, KS distances, n_eq counts

Usage:
    python scripts/analyze_dataset.py \
        --dataset-dir artifacts/datasets/validation \
        --output-dir  artifacts/validation/_analysis/dataset_analysis/validation
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from dataset_class import LTLDataset

from scripts._dataset_analysis import plots, shape_uniformity
from scripts._dataset_analysis.metrics import (
    ALL_OPERATORS,
    n_proposition_occurrences,
    n_unique_propositions,
    operator_counts,
    shape_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mc-shape-n", type=int, default=100_000,
                   help="Boltzmann samples per depth for the P1 reference.")
    p.add_argument("--similarity-pairs-per-depth", type=int, default=50_000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-shape-uniformity", action="store_true")
    p.add_argument("--no-embeddings", action="store_true",
                   help="Skip the PCA scatter (e.g. if embeddings are huge).")
    p.add_argument("--mmap-satisfactions", action="store_true",
                   help="Load satisfactions.pt via mmap (avoids loading the full tensor "
                        "into RAM; required for large datasets with >~50 GB tensors).")
    return p.parse_args()


def _sat_rates_chunked(t: torch.Tensor, chunk_rows: int = 500) -> torch.Tensor:
    """Compute per-formula satisfaction rates without holding the full tensor in RAM.

    Processes ``chunk_rows`` rows at a time so peak allocation is
    ``chunk_rows × N_traces`` rather than ``N_formulas × N_traces``.
    """
    n, n_traces = t.shape
    sat = torch.empty(n, dtype=torch.float32)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        sat[start:end] = t[start:end].sum(dim=1).float() / n_traces
    return sat


def build_per_formula_dataframe(
    dataset: LTLDataset,
    sat: torch.Tensor | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for i, formula in enumerate(dataset.formulas):
        sm = shape_metrics(formula)
        ops = operator_counts(formula)
        row = {
            "formula_id": i,
            "depth": sm["depth"],
            "n_nodes": sm["n_nodes"],
            "n_leaves": sm["n_leaves"],
            "n_unary_internal": sm["n_unary_internal"],
            "n_binary_internal": sm["n_binary_internal"],
            "branching_ratio": sm["branching_ratio"],
            "mean_branch_imbalance": sm["mean_branch_imbalance"],
            "longest_path_concentration": sm["longest_path_concentration"],
            "leaf_depth_variance": sm["leaf_depth_variance"],
            "n_unique_props": n_unique_propositions(formula),
            "n_prop_occurrences": n_proposition_occurrences(formula),
            "length_chars": len(str(formula)),
            "satisfaction_rate": float(sat[i]) if sat is not None else float("nan"),
        }
        for op in ALL_OPERATORS:
            row[f"n_{op}"] = ops[op]
            row[f"present_{op}"] = ops[op] > 0
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_csv(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for d, sub in df.groupby("depth"):
        summary_rows.append({
            "depth": int(d),
            "n_formulas": int(len(sub)),
            "length_chars_mean": float(sub["length_chars"].mean()),
            "length_chars_median": float(sub["length_chars"].median()),
            "length_chars_p95": float(sub["length_chars"].quantile(0.95)),
            "length_chars_p99": float(sub["length_chars"].quantile(0.99)),
            "length_chars_max": int(sub["length_chars"].max()),
            "n_nodes_mean": float(sub["n_nodes"].mean()),
            "branching_ratio_mean": float(sub["branching_ratio"].mean(skipna=True)),
            "mean_branch_imbalance_mean": float(sub["mean_branch_imbalance"].mean(skipna=True)),
            "longest_path_concentration_mean": float(sub["longest_path_concentration"].mean()),
            "leaf_depth_variance_mean": float(sub["leaf_depth_variance"].mean()),
            "satisfaction_rate_mean": float(sub["satisfaction_rate"].mean(skipna=True)),
            "n_unique_props_mean": float(sub["n_unique_props"].mean()),
        })
    return pd.DataFrame(summary_rows).sort_values("depth").reset_index(drop=True)


def serialise_uniformity(uniformity: dict) -> dict:
    """Make the shape-uniformity result JSON-serialisable."""
    out = {
        "depths": uniformity["depths"],
        "n_eq": {str(d): int(n) for d, n in uniformity["n_eq"].items()},
        "shape_entropy_ratio": {
            str(d): (None if v is None else float(v))
            for d, v in uniformity["shape_entropy_ratio"].items()
        },
        "ks_distances": {
            str(d): {m: {"D": float(v[0]), "p": float(v[1])} for m, v in mdict.items()}
            for d, mdict in uniformity["ks_distances"].items()
        },
    }
    # Rank plot info: keep top-20 per depth (full list could be huge)
    rank = {}
    for d, rank_data in uniformity["shape_rank"].items():
        rank[str(d)] = {
            "empirical_top20": [
                {"shape_repr": repr(sh), "p": float(p)}
                for sh, p in rank_data["empirical"][:20]
            ],
            "reference_top20": [
                {"shape_repr": repr(sh), "p": float(p)}
                for sh, p in rank_data["reference"][:20]
            ],
        }
    out["shape_rank_top20"] = rank
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze_dataset] loading dataset from {args.dataset_dir}", file=sys.stderr)
    dataset = LTLDataset.load(
        args.dataset_dir,
        load_satisfactions=True,
        satisfactions_mmap=args.mmap_satisfactions,
    )
    print(f"[analyze_dataset] loaded {len(dataset)} formulas", file=sys.stderr)

    # Compute satisfaction rates in chunks to avoid holding the full tensor in RAM,
    # then release the mmap so the OS can reclaim those pages before the plots run.
    sat: torch.Tensor | None = None
    if dataset.satisfactions is not None:
        print("[analyze_dataset] computing satisfaction rates (chunked)...", file=sys.stderr)
        sat = _sat_rates_chunked(dataset.satisfactions)
        dataset.satisfactions = None
        gc.collect()

    df = build_per_formula_dataframe(dataset, sat=sat)
    df.to_csv(out / "per_formula.csv", index=False)
    print(f"[analyze_dataset] wrote {out / 'per_formula.csv'}", file=sys.stderr)

    summary = build_summary_csv(df)
    summary.to_csv(out / "dataset_summary.csv", index=False)
    print(f"[analyze_dataset] wrote {out / 'dataset_summary.csv'}", file=sys.stderr)

    uniformity = None
    if not args.no_shape_uniformity:
        print(f"[analyze_dataset] running shape uniformity diagnostics (mc_n={args.mc_shape_n})", file=sys.stderr)
        uniformity = shape_uniformity.compute_shape_uniformity(
            dataset.formulas, mc_n=args.mc_shape_n, rng_seed=args.rng_seed,
        )
        with open(out / "shape_uniformity.json", "w") as f:
            json.dump(serialise_uniformity(uniformity), f, indent=2)
        print(f"[analyze_dataset] wrote {out / 'shape_uniformity.json'}", file=sys.stderr)

    if args.no_figures:
        print("[analyze_dataset] --no-figures set, exiting", file=sys.stderr)
        return

    print("[analyze_dataset] rendering figures...", file=sys.stderr)
    plots.plot_depth_distribution(df, fig_dir / "depth_distribution")
    plots.plot_length_distribution(df, fig_dir / "length_distribution")
    plots.plot_length_by_depth_box(df, fig_dir / "length_by_depth_box")
    plots.plot_proposition_count_distribution(df, fig_dir / "proposition_count_distribution")
    plots.plot_depth_length_heatmap(df, fig_dir / "depth_length_2d_heatmap")
    if sat is not None:
        plots.plot_satisfaction_rate_by_depth(df, fig_dir / "satisfaction_rate_by_depth_violin")
        # Reload the satisfactions tensor fresh (previous mmap was released above)
        # so the similarity plot starts with the full memory budget.
        _sat_ds = LTLDataset.load(
            args.dataset_dir, load_satisfactions=True, satisfactions_mmap=True,
        )
        if _sat_ds.satisfactions is not None:
            plots.plot_within_depth_satisfaction_similarity(
                df,
                _sat_ds.satisfactions,
                fig_dir / "within_depth_satisfaction_similarity",
                pairs_per_depth=args.similarity_pairs_per_depth,
                rng_seed=args.rng_seed,
            )
        del _sat_ds
        gc.collect()
    plots.plot_operator_frequency_overall(df, fig_dir / "operator_frequency_overall")
    plots.plot_operator_frequency_by_depth(df, fig_dir / "operator_frequency_by_depth")
    plots.plot_operator_cooccurrence_heatmap(df, fig_dir / "operator_cooccurrence_heatmap")

    ref = uniformity["reference_metric_arrays"] if uniformity else None
    plots.plot_node_count_by_depth(df, ref, fig_dir / "node_count_by_depth_violin")
    plots.plot_branching_ratio_by_depth(df, ref, fig_dir / "branching_ratio_by_depth_violin")
    plots.plot_mean_branch_imbalance_by_depth(df, ref, fig_dir / "mean_branch_imbalance_by_depth_violin")
    plots.plot_longest_path_concentration_by_depth(df, ref, fig_dir / "longest_path_concentration_by_depth_violin")
    plots.plot_leaf_depth_variance_by_depth(df, ref, fig_dir / "leaf_depth_variance_by_depth_violin")

    if not args.no_embeddings and dataset.embeddings is not None:
        plots.plot_embedding_pca(df, dataset.embeddings.cpu().numpy(), fig_dir / "embedding_pca_by_depth")

    if uniformity is not None:
        plots.plot_shape_entropy_ratio(uniformity, fig_dir / "shape_entropy_ratio_by_depth")
        for d in uniformity["shape_rank"].keys():
            plots.plot_shape_rank(uniformity, d, fig_dir / f"shape_rank_plot_d{d}")
        plots.plot_ks_distance_heatmap(uniformity, fig_dir / "ks_distance_per_metric_by_depth")

    print(f"[analyze_dataset] done. figures under {fig_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
