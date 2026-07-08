import json
import os
from typing import Any

import torch
from formula_class import eval_traces_batch, Formula, Atom, And, Next, Not
from formula_utils import sample_traces, sample_traces_correlated, sample_formulas, str_to_formula

class LTLKernel:
    def __init__(self, T: int, AP: int, time_index: int = 0, device: str | None = None, seed: int | None = None):
        """
        Kernel for LTL formulas based on sampled traces.

        - T: Maximum trace length.
        - AP: Number of atomic propositions.
        - seed: Specifies the seed used in each of the random number generators, for reproducability.
        """
        self.T: int                                 = T
        self.AP: int                                = AP
        self.seed: int | None                       = seed

        if device is not None:
            resolved_device = str(torch.device(device))
        else:
            resolved_device = ('cuda'
                               if torch.cuda.is_available()
                               else 'mps'
                               if torch.backends.mps.is_available()
                               else 'cpu')

        self.device: str = resolved_device

        if time_index > T-1 or time_index < 0:
            raise ValueError(f'The specified time_index has to fall in the range (0,...{T-1}).')
        else:
            self.time_index = time_index
        
        self.rng: torch.Generator = (torch.Generator(device=self.device).manual_seed(self.seed)
                                     if self.seed is not None
                                     else torch.Generator(device=self.device))
        
        self.anchor_formulas: list[Formula]         = []            # list of anchor formulae
        self.traces: torch.Tensor | None            = None          # (N, AP, T), bool, Tensor
        self.m: int | None                          = None          # number of anchor formula
        self.F: torch.Tensor | None                 = None          # feature matrix (m, N), ±1, Tensor


    def set_device(self, device: str | torch.device, non_blocking: bool = True) -> None:
        """Move kernel tensors and RNG to the specified device when needed."""
        target_device = torch.device(device)
        target_device_str = str(target_device)

        rng_state = self.rng.get_state().cpu() if self.rng is not None else None

        if self.traces is not None and self.traces.device != target_device:
            self.traces = self.traces.to(device=target_device, non_blocking=non_blocking)

        if self.F is not None and self.F.device != target_device:
            self.F = self.F.to(device=target_device, non_blocking=non_blocking)

        if self.device != target_device_str:
            self.device = target_device_str

        if rng_state is not None:
            self.rng = torch.Generator(device=self.device)
            self.rng.set_state(rng_state)
        elif self.rng is None:
            self.rng = torch.Generator(device=self.device)
            if self.seed is not None:
                self.rng.manual_seed(self.seed)



    # ----------- Sampling -----------
    def sample_traces_kernel(self, N: int) -> torch.Tensor:
        """
        Method for adding a random sample of traces to the kernel.
        - N: specifies the number of sampled traces.
        Implicit arguments are, AP, T, seed:
        - AP: specifies the number of atomic propositions in each trace.
        - T: specifies the length of each of the sampled traces.
        - rng: specifies the random number generator used, for reproducibility.
        """
        self.traces = sample_traces(N,
                                    n_ap=self.AP,
                                    trace_length=self.T,
                                    rng=self.rng,
                                    device=self.device)


    def sample_traces_kernel_correlated(self, N: int, low_variance_ratio: float = 0.5, low_var_switch_prob: float = 0.1) -> torch.Tensor:
        """Sample a mixture of low-variance and high-variance traces with automatic deduplication."""
        self.traces = sample_traces_correlated(N,
                                               n_ap=self.AP,
                                               trace_length=self.T,
                                               rng=self.rng,
                                               device=self.device,
                                               low_variance_ratio=low_variance_ratio,
                                               low_var_switch_prob=low_var_switch_prob)



    def add_anchor_formulas(self, formulas: list):
        """
        Method for manually adding (a list of) formulae.
        """
        self.anchor_formulas.extend(formulas)
        self.m = len(self.anchor_formulas)



    def construct_anchor_formulas_kernel(self):
        """
        Method for constructing the set anchor formulae.

        """

        literal_cache = {
            atom_idx: Atom(atom_idx)
            for atom_idx in range(self.AP)
        }

        Chi: list[Formula] = []

        for t in range(self.T):
            for i in range (self.AP):
                formula = literal_cache[i]
                for _ in range(t): 
                    formula = Next(formula)
                Chi.append(formula)
                

        self.add_anchor_formulas(Chi)



    def sample_anchor_formulas_kernel(self, m: int = 1024, p_leaf_range: float = (0.4,0.6), max_depth: int = 6, force_tree: bool = True):
        """
        Method for adding a random sample of formulae to the kernel.
        - m: specifies the number of sampled formulae.
        - p_leaf_range: (Default = (0.4,0.6)) specifies the odds of each node being a leaf. Higher probability reduces average (bounded) formula complexity.
        - max_depth: (Default = 6) specifies the maximum formula complexity.
        - force_tree: (Default = True) forces the root of the syntax tree to be an operator. Without this, p_leaf_range percent of the sample will be just an AP.

        Implicit arguments are: AP, T, seed.
        - AP: specifies the number of atomic propositions available to each formula.
        - rng: specifies the random number generator used, for reproducibility.
        """
        sample = sample_formulas(n_formula=m,
                                 p_leaf_range=p_leaf_range,
                                 max_depth=max_depth,
                                 n_ap=self.AP,
                                 force_tree=force_tree,
                                 rng=self.rng,
                                 device=self.device)

        self.add_anchor_formulas(sample)



    def sample_anchor_formulas_kernel_cosine_controlled(self, m: int = 1024, p_leaf_range: tuple[float, float] = (0.4, 0.6), max_depth: int = 6, force_tree: bool = True, batch_size: int = 512, threshold: float = 0.8, max_attempts_per_formula: int = 100):
        """
        Rejection-sample ``m`` anchor formulae under a symmetric Hamming-band constraint.

        Anchors are selected under the signed-dot kernel sim^pm(psi, psi_i) = k^pm/N, the
        normalized dot product of the +/-1 satisfaction vectors (2*satvec - 1). Because
        k^pm = N - 2*D_H, we have sim^pm = 1 - 2*D_H/N, so bounding ``|sim^pm| <= threshold``
        is exactly a symmetric Hamming band (1-threshold)/2 <= D_H/N <= (1+threshold)/2: a
        candidate is rejected if it is too *similar* (near-duplicate) OR too *anti-similar*
        (near-complement / negation) to any accepted anchor. Trivial candidates (tautologies
        / contradictions, i.e. constant satvecs) are rejected outright -- their centered
        feature row is 0, so they contribute nothing to any embedding under the covariance
        kernel and are degenerate anchors.

        - m: number of anchors to accept.
        - p_leaf_range: leaf-probability range passed to the formula sampler.
        - max_depth: maximum syntax-tree depth for candidates.
        - force_tree: force the candidate root to be an operator.
        - batch_size: batch size for evaluating a candidate's satvec on the traces.
        - threshold: similarity band half-width tau in (0, 1).
        - max_attempts_per_formula: attempts before giving up on the next anchor.

        Implicit arguments are AP, T, seed (via self.rng), for reproducibility.
        """

        if self.traces is None:
            raise ValueError('Please sample traces before calling sample_anchor_formulas_kernel_cosine_controlled so similarity can be computed.')

        N = self.traces.size(dim=0)
        if N == 0:
            raise ValueError('Traces tensor is empty, cannot evaluate similarity.')

        sqrt_N = float(N) ** 0.5
        # Preallocated matrix of accepted +/-1 vectors scaled by 1/sqrt(N); one accepted anchor per row.
        accepted_norm = torch.empty((m, N), dtype=torch.float32, device=self.device)
        selected_formulas: list[Formula] = []
        n_accepted = 0
        rejected_trivial = 0
        rejected_similar = 0

        for idx in range(m):
            attempts = 0
            while attempts < max_attempts_per_formula:
                attempts += 1
                candidate = sample_formulas(n_formula=1,
                                            p_leaf_range=p_leaf_range,
                                            max_depth=max_depth,
                                            n_ap=self.AP,
                                            force_tree=force_tree,
                                            rng=self.rng,
                                            device=self.device)[0]

                sats = self._evaluate_formula_on_traces(formula=candidate, batch_size=batch_size)  # (N,) bool

                # Reject trivial candidates (constant satvec -> zero embedding, degenerate anchor).
                if bool(torch.all(sats)) or not bool(torch.any(sats)):
                    rejected_trivial += 1
                    continue

                candidate_norm = (sats.to(dtype=torch.float32) * 2.0 - 1.0) / sqrt_N  # +/-1 signed vector / sqrt(N)

                if n_accepted > 0:
                    sims = accepted_norm[:n_accepted] @ candidate_norm  # (n_accepted,) = sim^pm to each accepted anchor
                    if float(sims.abs().max()) > threshold:             # symmetric band: reject near-duplicate OR near-complement
                        rejected_similar += 1
                        continue

                accepted_norm[n_accepted] = candidate_norm
                selected_formulas.append(candidate)
                n_accepted += 1
                break
            else:
                raise RuntimeError(
                    f'Unable to sample a sufficiently distinct, non-trivial formula after '
                    f'{max_attempts_per_formula} attempts for anchor index {idx} '
                    f'(rejected_trivial={rejected_trivial}, rejected_similar={rejected_similar}).'
                )

        print(
            f'Accepted {n_accepted} anchors at threshold tau={threshold} '
            f'(rejected_trivial={rejected_trivial}, rejected_similar={rejected_similar}).'
        )
        self.add_anchor_formulas(selected_formulas)



    # ----------- Evaluation -----------
    def build_F(self, batch_size: int = 512) -> torch.Tensor:
        """
        Method for building the feature matrix F from the sampled formulae and traces.
        - formulas: list of formulae length m.
        - all_traces: Tensor shape (N, AP, T), dtype=bool.
        Specifies self.F: 
        - F: Tensor of shape (m, N) with {0,1} values, dtype=int8.
        """
        if self.traces is None and self.anchor_formulas is []:
            raise ValueError('Please first sample traces and formulas, using the sample_traces(N) and sample_formulas() method respectively.')

        if not(self.traces is None) and self.anchor_formulas is []:
            raise ValueError('You have not yet sampled formulas. Please do so using the sample_formulas() method.')
        
        if self.traces is None and not(self.anchor_formulas is []):
            raise ValueError('You have not yet sampled traces. Please do so using the sample_traces() method.')
        

        N = self.traces.size(dim=0)
        m = len(self.anchor_formulas)
        F = torch.empty((m, N), dtype=torch.float32, device=self.device)
        for i, phi in enumerate(self.anchor_formulas):
            sats = self._evaluate_formula_on_traces(formula=phi,batch_size=batch_size)
            vals = torch.where(sats, 
                               torch.tensor(1.0, dtype=torch.float32, device=self.device),
                               torch.tensor(0.0, dtype=torch.float32, device=self.device))  # (B,)
            F[i,:] = vals

        self.F = F



    # ----------- Kernel eval helper -----------
    def _evaluate_formula_on_traces(self, formula: Formula, batch_size: int, time_index: int | None = None) -> torch.Tensor:
        """Return boolean satisfaction vector of length N for the provided formula."""
        if time_index is None:
            time_index = self.time_index

        if self.traces is None:
            raise ValueError('Please sample traces before evaluating formulas.')

        N = self.traces.size(dim=0)
        sats = torch.empty(N, dtype=torch.bool, device=self.device)
        j = 0
        while j < N:
            j1 = min(N, j + batch_size)
            batch = self.traces[j:j1]
            batch_sats = eval_traces_batch(formula, batch)
            sats[j:j1] = batch_sats[:, time_index]
            j = j1
        return sats


    def _covariance_embeddings(self, phi_sats_2d: torch.Tensor) -> torch.Tensor:
        """Exact, reproducible covariance-kernel embeddings for a batch of satvecs.

        Implements the base-rate form of the covariance kernel,
            k^cov(phi, psi_j) = A_j / N - (B_j * C) / N^2,
        where, over the N traces, A_j = |{psi_j and phi both hold}| (joint count),
        B_j = |{psi_j holds}| (anchor count) and C = |{phi holds}| (target count).
        This equals the centered dot product (F_centered @ phi_centered)/N exactly.

        All three counts are exact integers: the 0/1 matmul A = Phi @ F^T never rounds
        in float32 because every partial sum is an integer <= N < 2**24. The counts are
        combined in int64 and the ONLY floating-point operation is a single float64
        division by N^2, so the embedding is bit-identical on any IEEE-754 hardware,
        independent of BLAS / GPU / thread count. See kernel reproducibility notes.

        - phi_sats_2d: (B, N) tensor of {0,1}/bool satisfaction vectors.
        Returns: (B, m) float32 embeddings.
        """
        if self.F is None:
            raise ValueError("The Feature Matrix has not yet been built. Please do so using the build_F() method.")
        if self.traces is None:
            raise ValueError('Please sample traces before computing embeddings.')

        N = self.traces.size(dim=0)
        if N >= (1 << 24):
            raise ValueError(
                f"N={N} exceeds 2**24; the 0/1 count matmul would no longer be exact in float32."
            )

        Phi = phi_sats_2d.to(device=self.device, dtype=torch.float32)   # (B, N), values in {0, 1}
        joint = Phi @ self.F.t()                                        # (B, m) exact integer joint counts A
        target_counts = Phi.sum(dim=1)                                  # (B,)   exact integer counts C
        anchor_counts = self.F.sum(dim=1)                               # (m,)   exact integer counts B_j

        A = joint.to(torch.int64)
        C = target_counts.to(torch.int64).unsqueeze(1)                  # (B, 1)
        B = anchor_counts.to(torch.int64).unsqueeze(0)                  # (1, m)
        numerator = A * N - B * C                                       # (B, m) int64, exact (|.| <= N^2 < 2**53)
        emb = numerator.to(torch.float64) / float(N * N)               # single IEEE-754 division
        return emb.to(torch.float32)


    def _maybe_move_to_cpu(self, emb: torch.Tensor, move_to_cpu: bool) -> torch.Tensor:
        if move_to_cpu and self.device in ('cuda', 'mps'):
            emb = emb.cpu()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            else:
                torch.mps.empty_cache()
        return emb


    def _compute_embedding_from_sats(self, phi_sats: torch.Tensor, move_to_cpu: bool) -> torch.Tensor:
        emb = self._covariance_embeddings(phi_sats.reshape(1, -1)).squeeze(0)
        return self._maybe_move_to_cpu(emb, move_to_cpu)



    # ----------- Embedding Computation -----------
    def compute_formula_embedding(self, formula: Formula, batch_size: int = 512) -> torch.Tensor:
        """
        Method for computing the embedding of formula, from feature matrix F.
        - formula: the formula for which the embedding is to be calcualted.
        - batch size: (Default = 512) the size of the batches used during evaluation of the formula, adjustable for memory management.
        - time index: (Default = 0) the timepoint of the trace at which the formula is evaluated.
        Returns:
            - emb: Tensor (m), the embedding of formula, where m = len(self.anchor_formulas) the number of anchor formulae.
        """ 
        phi_sats = self._evaluate_formula_on_traces(formula=formula,batch_size=batch_size)
        return self._compute_embedding_from_sats(phi_sats,move_to_cpu=True)
    


    def compute_formula_embedding_no_move(self, formula: Formula, batch_size: int = 512) -> torch.Tensor:
        """
        Method for computing the embedding of formula, from feature matrix F.
        - formula: the formula for which the embedding is to be calcualted.
        - batch size: (Default = 512) the size of the batches used during evaluation of the formula, adjustable for memory management.
        - time index: (Default = 0) the timepoint of the trace at which the formula is evaluated.
        Returns:
            - emb: Tensor (m), the embedding of formula, where m = len(self.anchor_formulas) the number of anchor formulae.
        """ 
        phi_sats = self._evaluate_formula_on_traces(formula=formula,batch_size=batch_size)
        return self._compute_embedding_from_sats(phi_sats, move_to_cpu=False)


    def compute_embedding_from_satisfaction(self, phi_sats: torch.Tensor, move_to_cpu: bool = False) -> torch.Tensor:
        """Compute the kernel embedding directly from a boolean satisfaction vector."""
        return self._compute_embedding_from_sats(phi_sats, move_to_cpu=move_to_cpu)


    def compute_embeddings_from_satisfactions(self, phi_sats_2d: torch.Tensor, move_to_cpu: bool = False) -> torch.Tensor:
        """Exact, reproducible covariance embeddings for a batch of satisfaction vectors.

        Batched counterpart of :meth:`compute_embedding_from_satisfaction`; see
        :meth:`_covariance_embeddings`. Intended for recomputing dataset embeddings from
        stored satvecs. Pass modest batches: with N = |traces| large, the (B, N) input
        dominates memory (e.g. B in the low hundreds to ~1024 for N = 500k). Placement /
        cache management is left to the caller (see move_to_cpu note on the primitive).
        - phi_sats_2d: (B, N) {0,1}/bool tensor.
        Returns: (B, m) float32 embeddings.
        """
        emb = self._covariance_embeddings(phi_sats_2d)
        return self._maybe_move_to_cpu(emb, move_to_cpu)




    # ----------- Dataset Generation -----------
    def sample_dataset_formulas_kernel(self, k: int, p_leaf_range: float, max_depth: int, force_tree: bool = True):
        """
        Method for adding a random sample of formulae to the kernel.
        - k: specifies the number of sampled formulae.
        - p_leaf_range: (Default = (0.4,0.6)) specifies the odds of each node being a leaf. Higher probability reduces average (bounded) formula complexity.
        - max_depth: (Default = 6) specifies the maximum formula complexity.
        - force_tree: (Default = True) forces the root of the syntax tree to be an operator. Without this, p_leaf_range percent of the sample will be just an AP.

        Implicit arguments are: AP, T, seed.
        - AP: specifies the number of atomic propositions available to each formula.
        - rng: specifies the random number generator used, for reproducibility.
        """
        sample = sample_formulas(n_formula=k,
                                 p_leaf_range=p_leaf_range,
                                 max_depth=max_depth,
                                 n_ap=self.AP,
                                 force_tree=force_tree,
                                 rng=self.rng,
                                 device=self.device)

        return sample

    def num_formulas(self, max_depth):
        G = [self.AP]  # G(0)
        for d in range(1, max_depth + 1):
            unary = 4 * G[d-1]
            binary = 4 * sum(G[k] * G[d-1] for k in range(d))
            G.append(unary + binary)
        return sum(G)


    # ----------- Persistence -----------
    def save(self, dirpath: str) -> None:
        """Persist kernel hyperparameters, sampled structures and RNG state."""
        os.makedirs(dirpath, exist_ok=True)

        metadata: dict[str, Any] = {
            "T": self.T,
            "AP": self.AP,
            "seed": self.seed,
            "device": self.device,
            "time_index": self.time_index,
            "m": self.m,
            "has_traces": self.traces is not None,
            "has_F": self.F is not None,
        }

        metadata_path = os.path.join(dirpath, "metadata.json")
        anchor_path = os.path.join(dirpath, "anchor_formulas.jsonl")
        rng_state_path = os.path.join(dirpath, "rng_state.pt")

        # Anchor formulas (one per line for readability)
        with open(anchor_path, "w", encoding="utf-8") as fp:
            for formula in self.anchor_formulas:
                fp.write(str(formula) + "\n")
        metadata["anchor_formula_count"] = len(self.anchor_formulas)

        if self.traces is not None:
            torch.save(self.traces.detach().to("cpu"), os.path.join(dirpath, "traces.pt"))

        if self.F is not None:
            torch.save(self.F.detach().to("cpu"), os.path.join(dirpath, "F.pt"))

        if self.rng is not None:
            torch.save(self.rng.get_state().cpu(), rng_state_path)
            metadata["has_rng_state"] = True
        else:
            metadata["has_rng_state"] = False

        with open(metadata_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)


    @classmethod
    def load(cls, dirpath: str, device: str | None = None) -> "LTLKernel":
        """Restore a kernel that was saved via :meth:`save`."""
        metadata_path = os.path.join(dirpath, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No kernel metadata found at {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)
            

        if device is not None and device != metadata["device"]:
            resolved_device = device
        else:
            resolved_device = metadata["device"]

        kernel = cls(T=int(metadata["T"]), AP=int(metadata["AP"]), time_index=int(metadata["time_index"]), device=resolved_device, seed=metadata.get("seed"))

        # Anchor formulas
        anchor_path = os.path.join(dirpath, "anchor_formulas.jsonl")
        anchor_formulas: list[Formula] = []
        if os.path.exists(anchor_path):
            with open(anchor_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    text = line.strip()
                    if text:
                        anchor_formulas.append(str_to_formula(text))
        kernel.anchor_formulas = anchor_formulas
        kernel.m = len(anchor_formulas) if anchor_formulas else metadata.get("m")

        # Tensors
        traces_path = os.path.join(dirpath, "traces.pt")
        if metadata.get("has_traces") and os.path.exists(traces_path):
            kernel.traces = torch.load(traces_path, map_location=kernel.device)

        F_path = os.path.join(dirpath, "F.pt")
        if metadata.get("has_F") and os.path.exists(F_path):
            kernel.F = torch.load(F_path, map_location=kernel.device)

        # RNG state (if present)
        rng_state_path = os.path.join(dirpath, "rng_state.pt")
        if metadata.get("has_rng_state") and os.path.exists(rng_state_path):
            state_tensor = torch.load(rng_state_path, map_location="cpu")
            kernel.rng = torch.Generator(device=kernel.device)
            kernel.rng.set_state(state_tensor)

        return kernel