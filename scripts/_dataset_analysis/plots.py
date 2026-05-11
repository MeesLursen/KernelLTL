"""Dataset-analysis figures.

Every figure is a self-contained ``plot_*`` function that takes the
specific inputs it needs and writes ``<stem>.pdf`` and ``<stem>.png``.
Style: seaborn ``whitegrid`` with the ``colorblind`` palette.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import ALL_OPERATORS

sns.set_theme(style="whitegrid", palette="colorblind")


def _save(fig: plt.Figure, stem: Path, dpi: int = 200) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Distribution of basic per-formula stats
# ---------------------------------------------------------------------------


def plot_depth_distribution(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["depth"].value_counts().sort_index()
    ax.bar(counts.index.astype(int), counts.values, color=sns.color_palette()[0])
    ax.set_xlabel("target_depth")
    ax.set_ylabel("formula count")
    ax.set_title("Formula count by depth")
    for x, y in zip(counts.index.astype(int), counts.values):
        ax.text(x, y, str(y), ha="center", va="bottom", fontsize=9)
    _save(fig, stem)


def plot_length_distribution(df: pd.DataFrame, stem: Path) -> None:
    depths = sorted(df["depth"].unique())
    n = len(depths)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4), sharey=True)
    if n == 0:
        axes = [axes]
    axes_list = list(np.atleast_1d(axes))
    sns.histplot(df["length_chars"], bins=40, ax=axes_list[0], color=sns.color_palette()[0])
    axes_list[0].set_title("Overall")
    axes_list[0].set_xlabel("length (chars)")
    for ax, d in zip(axes_list[1:], depths):
        sns.histplot(df.loc[df["depth"] == d, "length_chars"], bins=40, ax=ax, color=sns.color_palette()[0])
        ax.set_title(f"depth = {d}")
        ax.set_xlabel("length (chars)")
    fig.suptitle("Formula length distribution")
    fig.tight_layout()
    _save(fig, stem)


def plot_length_by_depth_box(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="depth", y="length_chars", ax=ax)
    ax.set_xlabel("target_depth")
    ax.set_ylabel("length (chars)")
    ax.set_title("Formula length by depth")
    _save(fig, stem)


def plot_proposition_count_distribution(df: pd.DataFrame, stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df["n_unique_props"], discrete=True, ax=axes[0], color=sns.color_palette()[0])
    axes[0].set_xlabel("unique propositions per formula")
    axes[0].set_ylabel("count")
    axes[0].set_title("Unique propositions per formula")
    sns.histplot(df["n_prop_occurrences"], discrete=True, ax=axes[1], color=sns.color_palette()[1])
    axes[1].set_xlabel("proposition occurrences per formula")
    axes[1].set_title("Proposition occurrences per formula")
    fig.tight_layout()
    _save(fig, stem)


def plot_depth_length_heatmap(df: pd.DataFrame, stem: Path) -> None:
    pivot = (
        df.assign(length_bin=pd.cut(df["length_chars"], bins=20))
        .groupby(["depth", "length_bin"], observed=True)
        .size()
        .reset_index(name="count")
    )
    pivot["length_bin"] = pivot["length_bin"].astype(str)
    grid = pivot.pivot(index="depth", columns="length_bin", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(grid, ax=ax, cmap="viridis", cbar_kws={"label": "count"})
    ax.set_xlabel("length (chars), binned")
    ax.set_ylabel("target_depth")
    ax.set_title("Joint distribution of depth and length")
    fig.tight_layout()
    _save(fig, stem)


def plot_satisfaction_rate_by_depth(df: pd.DataFrame, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df, x="depth", y="satisfaction_rate", inner="quartile", ax=ax)
    ax.set_xlabel("target_depth")
    ax.set_ylabel("mean satisfaction over kernel traces")
    ax.set_title("Per-formula satisfaction rate by depth")
    _save(fig, stem)


def _batch_agree(
    satisfactions,
    i_arr: np.ndarray,
    j_arr: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Pairwise mean Hamming agreement computed in batches.

    Accepts either a numpy array or a (mmap) torch.Tensor.  Batching caps
    peak allocation at ``2 × batch_size × N_traces`` bytes instead of
    ``2 × n_pairs × N_traces``.
    """
    import torch
    is_tensor = isinstance(satisfactions, torch.Tensor)
    agree = np.empty(len(i_arr), dtype=np.float32)
    for b in range(0, len(i_arr), batch_size):
        ib = i_arr[b:b + batch_size]
        jb = j_arr[b:b + batch_size]
        if is_tensor:
            ri = satisfactions[ib].float()
            rj = satisfactions[jb].float()
            agree[b:b + batch_size] = (ri == rj).float().mean(dim=1).numpy()
        else:
            ri = satisfactions[ib]
            rj = satisfactions[jb]
            agree[b:b + batch_size] = np.mean(ri == rj, axis=1)
    return agree


def plot_within_depth_satisfaction_similarity(
    df: pd.DataFrame,
    satisfactions,          # torch.Tensor (mmap) or np.ndarray
    stem: Path,
    *,
    pairs_per_depth: int = 50_000,
    rng_seed: int = 0,
    batch_size: int = 500,
) -> None:
    """Histogram of pairwise Hamming agreement on traces, per depth."""
    rng = np.random.default_rng(rng_seed)
    depths = sorted(df["depth"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = sns.color_palette(n_colors=len(depths))
    for d, color in zip(depths, palette):
        idxs = df.index[df["depth"] == d].to_numpy()
        if len(idxs) < 2:
            continue
        n_pairs = min(pairs_per_depth, len(idxs) * (len(idxs) - 1) // 2)
        i = rng.choice(idxs, n_pairs)
        j = rng.choice(idxs, n_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        if len(i) == 0:
            continue
        agree = _batch_agree(satisfactions, i, j, batch_size)
        sns.kdeplot(agree, ax=ax, label=f"d={d}", color=color, fill=False)
    ax.set_xlabel("pairwise Hamming agreement on traces")
    ax.set_ylabel("density")
    ax.set_title("Within-depth satisfaction similarity")
    ax.legend(title="depth")
    _save(fig, stem)


# ---------------------------------------------------------------------------
# 2. Operator structure
# ---------------------------------------------------------------------------


def plot_operator_frequency_overall(df: pd.DataFrame, stem: Path) -> None:
    counts = {op: df[f"n_{op}"].sum() for op in ALL_OPERATORS}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(counts.keys()), list(counts.values()), color=sns.color_palette()[0])
    ax.set_xlabel("operator")
    ax.set_ylabel("total occurrences across dataset")
    ax.set_title("Operator frequency (overall)")
    _save(fig, stem)


def plot_operator_frequency_by_depth(df: pd.DataFrame, stem: Path) -> None:
    depths = sorted(df["depth"].unique())
    rows = []
    for d in depths:
        sub = df[df["depth"] == d]
        for op in ALL_OPERATORS:
            rows.append({"depth": d, "operator": op, "mean_count": sub[f"n_{op}"].mean()})
    long = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=long, x="operator", y="mean_count", hue="depth", ax=ax)
    ax.set_xlabel("operator")
    ax.set_ylabel("mean occurrences per formula")
    ax.set_title("Operator frequency by depth")
    ax.legend(title="depth", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, stem)


def plot_operator_cooccurrence_heatmap(df: pd.DataFrame, stem: Path) -> None:
    presence = df[[f"present_{op}" for op in ALL_OPERATORS]].astype(int).to_numpy()
    n = len(ALL_OPERATORS)
    cooc = np.zeros((n, n))
    for i in range(n):
        col_i = presence[:, i]
        denom = col_i.sum()
        if denom == 0:
            continue
        for j in range(n):
            cooc[i, j] = (col_i & presence[:, j]).sum() / denom
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cooc,
        annot=True,
        fmt=".2f",
        ax=ax,
        cmap="viridis",
        xticklabels=ALL_OPERATORS,
        yticklabels=ALL_OPERATORS,
        cbar_kws={"label": "P(col present | row present)"},
    )
    ax.set_title("Operator co-occurrence (conditional)")
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# 3. Tree shape (with reference overlay from Boltzmann sampler)
# ---------------------------------------------------------------------------


def _shape_metric_violin(
    df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    stem: Path,
    *,
    reference_arrays: dict[int, dict[str, list[float]]] | None = None,
) -> None:
    """Violin per depth for ``column``; overlay P1 reference if provided."""
    rows = []
    for _, r in df.iterrows():
        v = r[column]
        if pd.isna(v):
            continue
        rows.append({"depth": int(r["depth"]), "value": float(v), "source": "actual"})
    if reference_arrays is not None:
        for d, arrs in reference_arrays.items():
            for v in arrs.get(column, []):
                rows.append({"depth": int(d), "value": float(v), "source": "P1 reference"})
    if not rows:
        return
    long = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=long,
        x="depth",
        y="value",
        hue="source",
        split=True if reference_arrays is not None else False,
        inner="quartile",
        ax=ax,
    )
    ax.set_xlabel("target_depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if reference_arrays is None:
        ax.legend_.remove() if ax.legend_ else None
    _save(fig, stem)


def plot_node_count_by_depth(df, ref, stem):
    _shape_metric_violin(df, "n_nodes", "Node count by depth", "n_nodes", stem, reference_arrays=ref)


def plot_branching_ratio_by_depth(df, ref, stem):
    _shape_metric_violin(df, "branching_ratio", "Branching ratio by depth (binary / internal)", "branching_ratio", stem, reference_arrays=ref)


def plot_mean_branch_imbalance_by_depth(df, ref, stem):
    _shape_metric_violin(df, "mean_branch_imbalance", "Mean branch imbalance by depth (headline)", "mean_branch_imbalance", stem, reference_arrays=ref)


def plot_longest_path_concentration_by_depth(df, ref, stem):
    _shape_metric_violin(df, "longest_path_concentration", "Longest-path concentration by depth", "(longest_path_len + 1) / n_nodes", stem, reference_arrays=ref)


def plot_leaf_depth_variance_by_depth(df, ref, stem):
    _shape_metric_violin(df, "leaf_depth_variance", "Leaf depth variance by depth", "var(root-to-leaf path lengths)", stem, reference_arrays=ref)


# ---------------------------------------------------------------------------
# 4. Embedding coverage
# ---------------------------------------------------------------------------


def plot_embedding_pca(df: pd.DataFrame, embeddings: np.ndarray, stem: Path) -> None:
    """2D PCA scatter via numpy SVD, colored by depth."""
    if embeddings.shape[0] != len(df):
        raise ValueError(f"embedding rows ({embeddings.shape[0]}) != df rows ({len(df)})")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    # Take leading two right singular vectors. SVD of a (N, D) matrix:
    # we want the top components in feature-space; numpy SVD gives U S Vh.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vh[:2].T
    plot_df = df.copy()
    plot_df["pc1"] = proj[:, 0]
    plot_df["pc2"] = proj[:, 1]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="depth", palette="colorblind", s=8, alpha=0.5, ax=ax)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Kernel-induced embeddings (PCA), colored by depth")
    ax.legend(title="depth", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# 5. Shape-uniformity diagnostics
# ---------------------------------------------------------------------------


def plot_shape_entropy_ratio(uniformity: dict, stem: Path) -> None:
    depths = uniformity["depths"]
    ratios = [uniformity["shape_entropy_ratio"][d] for d in depths]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(depths, ratios, color=sns.color_palette()[2])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="uniform reference")
    ax.set_xlabel("target_depth")
    ax.set_ylabel("H(empirical) / log(N(d))")
    ax.set_ylim(0, 1.05)
    ax.set_title("Shape entropy ratio by depth")
    ax.legend()
    _save(fig, stem)


def plot_shape_rank(uniformity: dict, depth: int, stem: Path) -> None:
    if depth not in uniformity["shape_rank"]:
        return
    rank_data = uniformity["shape_rank"][depth]
    emp_ranked = rank_data["empirical"]
    ref_ranked = rank_data["reference"]

    p_emp = np.array([r[1] for r in emp_ranked])
    p_ref = np.array([r[1] for r in ref_ranked])

    n_total = uniformity.get("n_eq", {}).get(depth, 0)
    n_formulas = uniformity.get("n_formulas", {}).get(depth, 0)
    n_seen_emp = len(p_emp)
    n_seen_ref = len(p_ref)

    # x-axis spans 0 to n_formulas (the actual sample size).
    x_max = n_formulas if n_formulas > 0 else max(n_seen_emp, n_seen_ref)
    xs = np.arange(x_max)
    # Pad with NaN so each curve visibly stops at its own n_seen on log scale.
    emp_y = np.full(x_max, np.nan)
    emp_y[:min(n_seen_emp, x_max)] = p_emp[:x_max]
    ref_y = np.full(x_max, np.nan)
    ref_y[:min(n_seen_ref, x_max)] = p_ref[:x_max]

    coverage_emp = n_seen_emp / n_total if n_total else float("nan")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, emp_y, label=f"empirical (dataset, {n_seen_emp:,} distinct shapes)",
            color=sns.color_palette()[0])
    ax.plot(xs, ref_y,
            label=f"Boltzmann uniform sample, N={n_formulas:,} ({n_seen_ref:,} distinct)",
            color=sns.color_palette()[2], linestyle="--")
    ax.set_xlabel(f"shape rank (descending pmf, capped at n_formulas = {x_max:,})")
    ax.set_ylabel("probability")
    ax.set_title(
        f"Shape pmf rank plot — depth = {depth}\n"
        f"N_eq(d) = {n_total:,} | dataset shapes = {n_seen_emp:,} "
        f"({coverage_emp:.2%} of N_eq) | reference shapes = {n_seen_ref:,}"
    )
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, stem)


def plot_ks_distance_heatmap(uniformity: dict, stem: Path) -> None:
    depths = uniformity["depths"]
    metrics = list(next(iter(uniformity["ks_distances"].values())).keys()) if uniformity["ks_distances"] else []
    if not metrics or not depths:
        return
    grid = np.zeros((len(metrics), len(depths)))
    for j, d in enumerate(depths):
        for i, m in enumerate(metrics):
            v = uniformity["ks_distances"][d].get(m, (np.nan, np.nan))[0]
            grid[i, j] = v
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        grid,
        annot=True,
        fmt=".3f",
        cmap="rocket",
        xticklabels=depths,
        yticklabels=metrics,
        ax=ax,
        cbar_kws={"label": "KS distance vs P1 reference"},
    )
    ax.set_xlabel("target_depth")
    ax.set_title("KS distance: empirical vs uniform-arity-topology reference")
    fig.tight_layout()
    _save(fig, stem)
