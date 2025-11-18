import torch
from formula_class import eval_traces_batch, Formula, Atom, And, Next, Not
from formula_utils import sample_traces, sample_formulas

class LTLKernel:
    def __init__(self, T: int, AP: int, seed: int | None = None):
        """
        Kernel for LTL formulas based on sampled traces.

        - T: Maximum trace length.
        - AP: Number of atomic propositions.
        - seed: Specifies the seed used in each of the random number generators, for reproducability.
        """
        self.T: int                                 = T
        self.AP: int                                = AP
        self.seed: int | None                       = seed

        self.device: str = ('cuda'
                            if torch.cuda.is_available()
                            else 'mps'
                            if torch.backends.mps.is_available()
                            else 'cpu')
        
        self.rng: torch.Generator = (torch.Generator(device=self.device).manual_seed(self.seed)
                                     if self.seed is not None
                                     else torch.Generator(device=self.device))
        
        self.anchor_formulas: list[Formula]         = []            # list of anchor formulae
        self.traces: torch.Tensor | None            = None          # (N, AP, T), bool, Tensor
        self.m: int | None                          = None          # number of anchor formula
        self.F: torch.Tensor | None                 = None          # feature matrix (m, N), ±1, Tensor
        self.F_robustness: torch.Tensor | None      = None          # robustness feature matrix (m, N), Tensor
        self.K: torch.Tensor | None                 = None          # kernel matrix (m, m), Tensor
        self.K0: torch.Tensor | None                = None          # cosine kernel matrix (m, m), Tensor
        self.trace_atom_distances: torch.Tensor | None = None       # (AP, N, N) atom-wise Hamming distances



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

        # Reset cached structures that depend on the trace set
        self.F = None
        self.F_robustness = None
        self.trace_atom_distances = None

        self._precompute_atomwise_hamming()



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



    def sample_anchor_formulas_kernel(self, m: int = 1024, p_leaf: float = 0.5, max_depth: int = 6, force_tree: bool = True):
        """
        Method for adding a random sample of formulae to the kernel.
        - m: specifies the number of sampled formulae.
        - p_leaf: (Default = 0.5) specifies the odds of each node being a leaf. Higher probability reduces average (bounded) formula complexity.
        - max_depth: (Default = 6) specifies the maximum formula complexity.
        - force_tree: (Default = True) forces the root of the syntax tree to be an operator. Without this, p_leaf percent of the sample will be just an AP.

        Implicit arguments are: AP, T, seed.
        - AP: specifies the number of atomic propositions available to each formula.
        - rng: specifies the random number generator used, for reproducibility.
        """
        sample = sample_formulas(n_formula=m,
                                 p_leaf=p_leaf,
                                 max_depth=max_depth,
                                 n_ap=self.AP,
                                 force_tree=force_tree,
                                 rng=self.rng,
                                 device=self.device)

        self.add_anchor_formulas(sample)



    def sample_anchor_formulas_kernel_cosine_controlled(self, m: int = 1024, p_leaf: float = 0.5, max_depth: int = 6, force_tree: bool = True, batch_size = 512, max_attempts_per_formula = 100):
        """
        Method for adding a random sample of formulae to the kernel.
        - m: specifies the number of sampled formulae.
        - p_leaf: (Default = 0.5) specifies the odds of each node being a leaf. Higher probability reduces average (bounded) formula complexity.
        - max_depth: (Default = 6) specifies the maximum formula complexity.
        - force_tree: (Default = True) forces the root of the syntax tree to be an operator. Without this, p_leaf percent of the sample will be just an AP.

        Implicit arguments are: AP, T, seed.
        - AP: specifies the number of atomic propositions available to each formula.
        - rng: specifies the random number generator used, for reproducibility.
        """
    
        if self.traces is None:
            raise ValueError('Please sample traces before calling sample_anchor_formulas_kernel2 so cosine similarity can be computed.')

        similarity_threshold = 0.8
        time_index = 0

        if time_index < 0 or time_index >= self.T:
            raise ValueError(f'time_index must be between 0 and {self.T - 1}, but received {time_index}.')

        N = self.traces.size(dim=0)
        if N == 0:
            raise ValueError('Traces tensor is empty, cannot evaluate cosine similarity.')

        one = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        zero = torch.tensor(-1.0, dtype=torch.float32, device=self.device)

        def _formula_trace_vector(phi: Formula) -> torch.Tensor:
            vals = torch.empty(N, dtype=torch.float32, device=self.device)
            j = 0
            while j < N:
                j1 = min(N, j + batch_size)
                batch = self.traces[j:j1]
                sats = eval_traces_batch(phi, batch)
                vals[j:j1] = torch.where(sats[:, time_index], one, zero)
                j = j1
            return vals

        selected_formulas: list[Formula]       = []
        normalized_vectors: list[torch.Tensor] = []

        for idx in range(m):
            attempts = 0
            while attempts < max_attempts_per_formula:
                attempts += 1
                candidate = sample_formulas(n_formula=1,
                                            p_leaf=p_leaf,
                                            max_depth=max_depth,
                                            n_ap=self.AP,
                                            force_tree=force_tree,
                                            rng=self.rng,
                                            device=self.device)[0]

                candidate_vec = _formula_trace_vector(candidate)
                denom = torch.linalg.norm(candidate_vec)
                if denom > 1e-8:
                    candidate_norm = candidate_vec / denom
                else:
                    candidate_norm = torch.zeros_like(candidate_vec)

                too_similar = False
                for prev_vec in normalized_vectors:
                    if torch.dot(candidate_norm, prev_vec).item() > similarity_threshold:
                        too_similar = True
                        break

                if too_similar:
                    continue

                selected_formulas.append(candidate)
                normalized_vectors.append(candidate_norm)
                break
            else:
                raise RuntimeError(f'Unable to sample a sufficiently distinct formula after {max_attempts_per_formula} attempts for index {idx}.')

        self.add_anchor_formulas(selected_formulas)



    # ----------- Evaluation -----------
    def build_F(self, batch_size: int = 512, time_index: int = 0) -> torch.Tensor:
        """
        Method for building the feature matrix F from the sampled formulae and traces.
        - formulas: list of formulae length m.
        - all_traces: Tensor shape (N, AP, T), dtype=bool.
        Specifies self.F: 
        - F: Tensor of shape (m, N) with ±1 values, dtype=int8.
        """
        if self.traces is None:
            raise ValueError('Please sample traces before building robustness features.')

        if not self.anchor_formulas:
            raise ValueError('Please add anchor formulas before building robustness features.')

        if self.trace_atom_distances is None:
            raise ValueError('Trace distances are unavailable. Ensure sample_traces_kernel has been called.')

        N = self.traces.size(dim=0)
        m = len(self.anchor_formulas)
        F_r = torch.empty((m, N), dtype=torch.float32, device=self.device)
        for i, phi in enumerate(self.anchor_formulas):
            F_r[i] = self._compute_formula_robustness_vector(phi, batch_size, time_index)

        self.F_robustness = F_r
        return self.F_robustness




    # ----------- Dataset Generation -----------
    def sample_dataset_formulas_kernel(self, k: int, p_leaf: float, max_depth: int, force_tree: bool = True):
        """
        Method for adding a random sample of formulae to the kernel.
        - k: specifies the number of sampled formulae.
        - p_leaf: (Default = 0.5) specifies the odds of each node being a leaf. Higher probability reduces average (bounded) formula complexity.
        - max_depth: (Default = 6) specifies the maximum formula complexity.
        - force_tree: (Default = True) forces the root of the syntax tree to be an operator. Without this, p_leaf percent of the sample will be just an AP.

        Implicit arguments are: AP, T, seed.
        - AP: specifies the number of atomic propositions available to each formula.
        - rng: specifies the random number generator used, for reproducibility.
        """
        sample = sample_formulas(n_formula=k,
                                 p_leaf=p_leaf,
                                 max_depth=max_depth,
                                 n_ap=self.AP,
                                 force_tree=force_tree,
                                 rng=self.rng,
                                 device=self.device)

        return sample



    # ----------- Embedding Computation -----------
    def compute_formula_embedding(self, formula: Formula, batch_size: int = 512, time_index: int = 0) -> torch.Tensor:
        """
        Method for computing the embedding of formula, from feature matrix F.
        - formula: the formula for which the embedding is to be calcualted.
        - batch size: (Default = 512) the size of the batches used during evaluation of the formula, adjustable for memory management.
        - time index: (Default = 0) the timepoint of the trace at which the formula is evaluated.
        Returns:
            - emb: Tensor (m), the embedding of formula, where m = len(self.anchor_formulas) the number of anchor formulae.
        """ 
        if self.F_robustness is None:
            raise ValueError('Robustness feature matrix has not been built. Call build_F_robustness first.')

        if self.traces is None:
            raise ValueError('Please sample traces before computing embeddings.')

        N = self.traces.size(dim=0)
        phi_rho = self._compute_formula_robustness_vector(formula, batch_size, time_index)
        emb = (self.F_robustness @ phi_rho) / float(N)

        if self.device == 'cuda':
            emb = emb.cpu()
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            emb = emb.cpu()
            torch.mps.empty_cache()

        return emb
    


    def compute_formula_embedding_no_move(self, formula: Formula, batch_size: int = 512, time_index: int = 0) -> torch.Tensor:
        """
        Method for computing the embedding of formula, from feature matrix F.
        - formula: the formula for which the embedding is to be calcualted.
        - batch size: (Default = 512) the size of the batches used during evaluation of the formula, adjustable for memory management.
        - time index: (Default = 0) the timepoint of the trace at which the formula is evaluated.
        Returns:
            - emb: Tensor (m), the embedding of formula, where m = len(self.anchor_formulas) the number of anchor formulae.
        """ 
        if self.F_robustness is None:
            raise ValueError('Robustness feature matrix has not been built. Call build_F_robustness first.')

        if self.traces is None:
            raise ValueError('Please sample traces before computing embeddings.')

        N = self.traces.size(dim=0)
        phi_rho = self._compute_formula_robustness_vector(formula, batch_size, time_index)
        emb = (self.F_robustness @ phi_rho) / float(N)

        return emb



    # ----------- Robustness Kernel Helpers -----------
    def _precompute_atomwise_hamming(self) -> None:
        """Precompute pairwise Hamming distances per atom across all traces."""
        if self.traces is None:
            raise ValueError('Traces must be sampled before precomputing distances.')

        N = self.traces.size(dim=0)
        if N == 0:
            raise ValueError('At least one trace is required to precompute distances.')

        traces_device = self.traces.to(self.device, dtype=torch.float32)
        pairwise = torch.empty((self.AP, N, N), dtype=torch.float32, device=self.device)
        for atom_idx in range(self.AP):
            atom_traces = traces_device[:, atom_idx, :]  # (N, T)
            dist = torch.cdist(atom_traces, atom_traces, p=1)  # (N, N)
            pairwise[atom_idx] = dist

        self.trace_atom_distances = pairwise


    def _evaluate_formula_on_traces(self, formula: Formula, batch_size: int, time_index: int) -> torch.Tensor:
        """Return boolean satisfaction vector of length N for the provided formula."""
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


    def _aggregate_atom_distances(self, atom_ids: list[int]) -> torch.Tensor:
        """Aggregate precomputed atom-wise distances for the provided atom indices."""
        if self.trace_atom_distances is None:
            raise ValueError('Trace distances have not been precomputed. Call sample_traces_kernel first.')

        if len(atom_ids) == 0:
            N = self.traces.size(dim=0)
            return torch.zeros((N, N), dtype=torch.float32, device=self.device)

        idx_tensor = torch.tensor(atom_ids, dtype=torch.long, device=self.trace_atom_distances.device)
        relevant = torch.index_select(self.trace_atom_distances, 0, idx_tensor)  # (k, N, N)
        summed = relevant.sum(dim=0)  # (N, N)
        return summed.to(self.device)


    def _compute_formula_robustness_vector(self, formula: Formula, batch_size: int, time_index: int) -> torch.Tensor:
        """Compute per-trace robustness scores for the provided formula."""
        if self.traces is None:
            raise ValueError('Please sample traces before computing robustness.')

        sats = self._evaluate_formula_on_traces(formula, batch_size, time_index)
        atom_ids = sorted(formula.atoms())
        relevant_distances = self._aggregate_atom_distances(atom_ids)  # (N, N)

        N = sats.size(dim=0)
        robustness = torch.zeros(N, dtype=torch.float32, device=self.device)
        pos_idx = torch.nonzero(sats, as_tuple=False).squeeze(1)
        neg_idx = torch.nonzero(~sats, as_tuple=False).squeeze(1)

        max_distance = float(self.T * len(atom_ids)) if atom_ids else 0.0

        if pos_idx.numel() > 0 and neg_idx.numel() > 0:
            pos_to_neg = relevant_distances.index_select(0, pos_idx)
            pos_to_neg = pos_to_neg.index_select(1, neg_idx)
            min_pos = pos_to_neg.min(dim=1).values
            robustness[pos_idx] = min_pos

            neg_to_pos = relevant_distances.index_select(0, neg_idx)
            neg_to_pos = neg_to_pos.index_select(1, pos_idx)
            min_neg = neg_to_pos.min(dim=1).values
            robustness[neg_idx] = -min_neg
        else:
            if pos_idx.numel() > 0:
                robustness[pos_idx] = max_distance
            if neg_idx.numel() > 0:
                robustness[neg_idx] = -max_distance

        return robustness