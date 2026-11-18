"""Exact geometry of the anchor span, shared by the geometry analysis and Experiment 2.

Everything both analyses measure about the representation is a statement about the
orthogonal projection P onto the span S of the anchors' centred feature vectors
(thesis Sec. "Anchor Set Selection"). This module computes it once, and exactly
wherever that is possible:

* Counts. Joint counts of 0/1 data (anchor x anchor, formula x anchor, formula x
  formula) come from fp32 matmuls. Every partial sum is an integer <= N < 2**24,
  so the counts are exact on any hardware; TF32 is switched off.
* Kernel values. Centring multiplies counts by N, so it is done in int64 and
  divided once in float64, as in ``LTLKernel._covariance_embeddings``.
* Algebra. Everything after the counts is float64.

The basis. W = X_A X_A^T is the Gram matrix of the anchors' feature vectors,
computed exactly from counts. Its eigenvalues above a tolerance read off the gap
in the spectrum give the rank r, and with W = U diag(lambda) U^T,

    M = U_r diag(lambda_r)^(-1/2)   (m x r),        Q = X_A^T M   (N x r),

the columns of Q are an orthonormal basis of S, so that P = Q Q^T (thesis,
Experiment 2 design). Every quantity is then a sum over the rows q_i of Q:

    leverage         l_i = ||q_i||^2
    total leverage   L_D = sum_{i in D} ||q_i||^2 = tr P_DD
    group leverage   l_D = lambda_max(sum_{i in D} q_i q_i^T) = lambda_max(P_DD)

and the squared length that a formula's reconstruction keeps is ||M^T e||^2, with
e its exact embedding (thesis eq. retention_from_embeddings).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Device and precision
# --------------------------------------------------------------------------- #

def set_exact_matmul() -> None:
    """IEEE fp32 matmuls (no TF32), on which the exactness of the 0/1 counts rests."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def resolve_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# Kernel inputs
# --------------------------------------------------------------------------- #

@dataclass
class KernelData:
    F: torch.Tensor           # (m, N) float32, values in {0, 1}, on the device
    traces: torch.Tensor      # (N, AP, T) bool, on the device
    anchors: list[str]        # canonical strings, row order of F
    T: int
    AP: int
    time_index: int

    @property
    def m(self) -> int:
        return int(self.F.shape[0])

    @property
    def N(self) -> int:
        return int(self.F.shape[1])


def load_kernel(kernel_dir: str | Path, device: torch.device) -> KernelData:
    """Read the saved kernel tensors directly (not through LTLKernel.load)."""
    kernel_dir = Path(kernel_dir)
    meta = json.loads((kernel_dir / "metadata.json").read_text())
    anchors = [line.strip() for line in open(kernel_dir / "anchor_formulas.jsonl") if line.strip()]
    F = torch.load(kernel_dir / "F.pt", map_location="cpu").to(torch.float32)
    traces = torch.load(kernel_dir / "traces.pt", map_location="cpu").to(torch.bool)
    if F.shape != (len(anchors), traces.shape[0]):
        raise ValueError(f"F is {tuple(F.shape)}, expected ({len(anchors)}, {traces.shape[0]})")
    if traces.shape[0] >= (1 << 24):
        raise ValueError("N >= 2**24: fp32 counts would no longer be exact")
    if not bool(((F == 0) | (F == 1)).all()):
        raise ValueError("F holds values other than 0 and 1")
    return KernelData(F=F.to(device), traces=traces.to(device), anchors=anchors,
                      T=int(meta["T"]), AP=int(meta["AP"]),
                      time_index=int(meta.get("time_index", 0)))


# --------------------------------------------------------------------------- #
# Exact counts and kernel values
# --------------------------------------------------------------------------- #

def exact_counts(S: torch.Tensor, F: torch.Tensor, *, row_chunk: int = 256
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Joint counts A = S F^T (n x m) and row counts C = S 1 (n,) as int64.

    ``S`` holds 0/1 satisfaction vectors as rows (bool or float), on F's device.
    """
    n = int(S.shape[0])
    A = np.empty((n, int(F.shape[0])), dtype=np.int64)
    C = np.empty(n, dtype=np.int64)
    for s in range(0, n, row_chunk):
        Sc = S[s:s + row_chunk].to(torch.float32)
        A[s:s + row_chunk] = torch.round(Sc @ F.T).to(torch.int64).cpu().numpy()
        C[s:s + row_chunk] = torch.round(Sc.sum(dim=1)).to(torch.int64).cpu().numpy()
    return A, C


def pairwise_counts(S: torch.Tensor, *, col_chunk: int = 50_000) -> np.ndarray:
    """Exact joint counts S S^T (n x n, int64) for 0/1 rows S, accumulated over traces."""
    n, N = int(S.shape[0]), int(S.shape[1])
    J = torch.zeros((n, n), dtype=torch.float64, device=S.device)
    for s in range(0, N, col_chunk):
        Sc = S[:, s:s + col_chunk].to(torch.float32)
        J += (Sc @ Sc.T).to(torch.float64)          # each chunk exact (sums <= col_chunk)
    return torch.round(J).to(torch.int64).cpu().numpy()


def embedding_numerators(A: np.ndarray, C: np.ndarray, B: np.ndarray, N: int) -> np.ndarray:
    """N^2 times the covariance embedding: A*N - C*B, exact in int64.

    e = numerators / N^2 equals k^cov(phi, psi_j) = A_j/N - (C/N)(B_j/N).
    """
    return A * np.int64(N) - np.outer(C, B)


def anchor_gram(F: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(W, J, B): the exact anchor Gram matrix in float64, its joint counts, anchor counts."""
    J, B = exact_counts(F, F)
    N = int(F.shape[1])
    W = embedding_numerators(J, B, B, N).astype(np.float64) / float(N) / float(N)
    return W, J, B


# --------------------------------------------------------------------------- #
# Spectrum, tolerance, basis
# --------------------------------------------------------------------------- #

def spectrum(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues in descending order and the matching eigenvectors (columns), float64."""
    lam, U = np.linalg.eigh(W)
    order = np.argsort(lam)[::-1]
    return lam[order], U[:, order]


def gap_tolerance(eigvals_desc: np.ndarray, *, min_gap_orders: float = 6.0) -> dict:
    """The tolerance below which an eigenvalue counts as zero, read off the gap.

    Eigenvalues at or below the numerical-rank floor lambda_max * m * eps are taken
    as zero. The gap between the smallest eigenvalue above the floor and the
    largest |eigenvalue| below it must span at least ``min_gap_orders`` orders of
    magnitude, otherwise the rank is ambiguous and this raises. The tolerance is
    the geometric midpoint of the gap.
    """
    lam = np.asarray(eigvals_desc, dtype=np.float64)
    m = lam.size
    floor = float(lam[0] * m * np.finfo(np.float64).eps)
    r = int(np.sum(lam > floor))
    if r == m:
        return {"tol": floor, "r": m, "floor": floor, "smallest_kept": float(lam[-1]),
                "largest_dropped_abs": None, "gap_orders": None, "rule": "full rank"}
    smallest_kept = float(lam[r - 1])
    largest_dropped = float(np.max(np.abs(lam[r:])))
    tiny = np.finfo(np.float64).tiny
    gap_orders = math.log10(smallest_kept / max(largest_dropped, tiny))
    if gap_orders < min_gap_orders:
        raise ValueError(f"no clear gap in the spectrum: {smallest_kept:.3e} kept vs "
                         f"{largest_dropped:.3e} dropped ({gap_orders:.1f} orders)")
    return {"tol": math.sqrt(smallest_kept * max(largest_dropped, tiny)), "r": r,
            "floor": floor, "smallest_kept": smallest_kept,
            "largest_dropped_abs": largest_dropped, "gap_orders": gap_orders,
            "rule": "geometric midpoint of the gap above the numerical floor"}


def sparse_null_basis(Z: np.ndarray, *, zero: float = 1e-9) -> np.ndarray:
    """Reduced row echelon form of the null-space basis Z (m x k), returned as (k x m).

    The eigenvectors of a repeated (zero) eigenvalue are an arbitrary rotation of the
    null space; the echelon form undoes it, so that dependencies among disjoint groups
    of anchors come out as separate vectors.
    """
    A = np.array(Z.T, dtype=np.float64)
    k, m = A.shape
    scale = np.abs(A).max() if A.size else 1.0
    row = 0
    for col in range(m):
        if row == k:
            break
        piv = row + int(np.argmax(np.abs(A[row:, col])))
        if abs(A[piv, col]) <= zero * scale:
            continue
        A[[row, piv]] = A[[piv, row]]
        A[row] /= A[row, col]
        for other in range(k):
            if other != row:
                A[other] -= A[other, col] * A[row]
        row += 1
    A[np.abs(A) <= zero] = 0.0
    return A


def integer_coefficients(v: np.ndarray, *, tol: float = 1e-6) -> np.ndarray | None:
    """Scale a dependency vector to small integers, or None if it is not rational."""
    nz = np.abs(v[v != 0])
    if nz.size == 0:
        return None
    for mult in (1, 2, 3, 4, 6, 12):
        c = v / nz.min() * mult
        if np.all(np.abs(c - np.round(c)) < tol):
            g = np.gcd.reduce(np.abs(np.round(c[c != 0])).astype(np.int64))
            return (np.round(c) / g).astype(np.int64)
    return None


class Projector:
    """The projection onto the anchor span, through the explicit basis Q (N x r)."""

    def __init__(self, F: torch.Tensor, B: np.ndarray, eigvals: np.ndarray, U: np.ndarray,
                 tol: float, *, chunk: int = 65_536):
        self.device = F.device
        self.m, self.N = int(F.shape[0]), int(F.shape[1])
        keep = eigvals > tol
        self.r = int(keep.sum())
        self.M_np = U[:, keep] / np.sqrt(eigvals[keep])                        # (m, r)
        self.M = torch.from_numpy(self.M_np).to(self.device)
        p = torch.from_numpy(np.asarray(B, dtype=np.float64) / self.N).to(self.device)
        pM = p @ self.M
        scale = 1.0 / math.sqrt(self.N)
        self.Q = torch.empty((self.N, self.r), dtype=torch.float64, device=self.device)
        for s in range(0, self.N, chunk):
            Fc = F[:, s:s + chunk].to(torch.float64)
            self.Q[s:s + chunk] = (Fc.T @ self.M - pM) * scale
        self.leverage = (self.Q * self.Q).sum(dim=1)                            # (N,)

    # --- checks -------------------------------------------------------------- #
    def orthonormality_error(self) -> float:
        G = self.Q.T @ self.Q
        G.diagonal().sub_(1.0)
        return float(G.abs().max())

    def centring_error(self) -> float:
        return float(self.Q.sum(dim=0).abs().max())

    # --- kept squared lengths ------------------------------------------------ #
    def kept_sq(self, numerators: np.ndarray, *, chunk: int = 2048) -> np.ndarray:
        """||M^T e||^2 per row, e = numerators / N^2 (exact embeddings or differences)."""
        n2 = float(self.N) * float(self.N)
        out = np.empty(numerators.shape[0], dtype=np.float64)
        for s in range(0, numerators.shape[0], chunk):
            E = torch.from_numpy(numerators[s:s + chunk].astype(np.float64) / n2).to(self.device)
            out[s:s + chunk] = ((E @ self.M) ** 2).sum(dim=1).cpu().numpy()
        return out

    # --- sets of traces ------------------------------------------------------ #
    def set_stats(self, D: torch.Tensor, *, want_rowsum: bool = False) -> dict:
        """Total leverage L_D and group leverage l_D of the traces D (1-D index tensor).

        l_D = lambda_max(Q_D^T Q_D). For |D| < r the |D| x |D| matrix Q_D Q_D^T has the
        same nonzero eigenvalues and is cheaper; for |D| > N/2 the complement is used,
        since Q_D^T Q_D = I - Q_{D^c}^T Q_{D^c}.
        """
        D = D.to(self.device)
        d = int(D.numel())
        out = {"size": d, "L": 0.0, "l": 0.0, "rowsum": None}
        if d == 0:
            return out
        out["L"] = float(self.leverage[D].sum())
        if d < self.r:
            QD = self.Q[D]
            out["l"] = float(torch.linalg.eigvalsh(QD @ QD.T)[-1])
        elif 2 * d > self.N:
            mask = torch.ones(self.N, dtype=torch.bool, device=self.device)
            mask[D] = False
            QC = self.Q[mask]
            G = torch.eye(self.r, dtype=torch.float64, device=self.device) - QC.T @ QC
            out["l"] = float(torch.linalg.eigvalsh(G)[-1])
        else:
            QD = self.Q[D]
            out["l"] = float(torch.linalg.eigvalsh(QD.T @ QD)[-1])
        if want_rowsum:
            out["rowsum"] = self.Q[D].sum(dim=0)
        return out


def ceiling(l_D: float, share: float) -> float:
    """kappa_D = min{1, l_D / (1 - |D|/N)} (thesis eq. retention_ceiling)."""
    return min(1.0, l_D / (1.0 - share)) if share < 1.0 else 1.0


def even_ceiling(r: int, share: float) -> float:
    """The even-division counterpart: l_D replaced by min{1, r |D|/N}."""
    return min(1.0, r * share / (1.0 - share)) if share < 1.0 else 1.0


# --------------------------------------------------------------------------- #
# Trace processes
# --------------------------------------------------------------------------- #

PROCESS_NAMES = ("iid", "markov", "baseline_all_false", "baseline_all_true", "mixed")


def flip_counts(traces: torch.Tensor) -> torch.Tensor:
    """Number of changes of truth value over all propositions and time steps, per trace."""
    return (traces[:, :, 1:] != traces[:, :, :-1]).sum(dim=(1, 2))


def flip_rule(flips: np.ndarray, *, n_ap: int, T: int, low_variance_ratio: float,
              switch_prob: float) -> tuple[np.ndarray, int, float]:
    """Bayes rule Markov vs i.i.d. from the flip count, with the mixing share as prior.

    Returns the decision per trace (True = Markov), the largest flip count classified
    as Markov, and the expected number of misclassified traces.
    """
    f = flips.astype(np.float64)
    steps = n_ap * (T - 1)
    log_markov = (math.log(low_variance_ratio) - n_ap * math.log(2)
                  + f * math.log(switch_prob) + (steps - f) * math.log(1 - switch_prob))
    log_iid = math.log(1 - low_variance_ratio) - n_ap * T * math.log(2)
    post = 1.0 / (1.0 + np.exp(log_iid - log_markov))
    decision = post > 0.5
    threshold = int(f[decision].max()) if decision.any() else -1
    return decision, threshold, float(np.minimum(post, 1 - post).sum())


def trace_processes(traces: torch.Tensor, origin_path: str | Path | None, *,
                    low_variance_ratio: float, switch_prob: float) -> tuple[np.ndarray, dict]:
    """Process code per trace (index into PROCESS_NAMES) and how it was obtained.

    From the replay labels (kernel_provenance/trace_origin.npy) when available; a
    stored row that is a baseline trace keeps the baseline label even if a Markov draw
    duplicated it. Otherwise from the flip-count rule, with the two constant traces
    identified by content.
    """
    N, n_ap, T = (int(x) for x in traces.shape)
    flat = traces.reshape(N, -1)
    all_false = (~flat).all(dim=1).cpu().numpy()
    all_true = flat.all(dim=1).cpu().numpy()
    codes = np.full(N, PROCESS_NAMES.index("mixed"), dtype=np.int8)
    if origin_path is not None and Path(origin_path).exists():
        bits = np.load(origin_path)
        if bits.shape != (N,):
            raise ValueError(f"{origin_path}: {bits.shape} labels for {N} traces")
        codes[bits == 8] = PROCESS_NAMES.index("iid")
        codes[bits == 4] = PROCESS_NAMES.index("markov")
        codes[(bits & 1) > 0] = PROCESS_NAMES.index("baseline_all_false")
        codes[(bits & 2) > 0] = PROCESS_NAMES.index("baseline_all_true")
        if not (all_false[codes == 2].all() and all_true[codes == 3].all()):
            raise ValueError("replay labels put a baseline label on a non-constant trace")
        info = {"source": "replay", "path": str(origin_path)}
    else:
        flips = flip_counts(traces).cpu().numpy()
        markov, threshold, expected = flip_rule(flips, n_ap=n_ap, T=T,
                                                low_variance_ratio=low_variance_ratio,
                                                switch_prob=switch_prob)
        codes[:] = np.where(markov, PROCESS_NAMES.index("markov"), PROCESS_NAMES.index("iid"))
        codes[all_false] = PROCESS_NAMES.index("baseline_all_false")
        codes[all_true] = PROCESS_NAMES.index("baseline_all_true")
        info = {"source": "flip_rule", "markov_iff_flips_at_most": threshold,
                "expected_misclassified_traces": expected}
    return codes, info
