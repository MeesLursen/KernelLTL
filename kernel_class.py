import torch
from formula_class import eval_traces_batch, Formula, Atom, And, Next, Not
from formula_utils import sample_traces, sample_traces_correlated, sample_formulas

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

        similarity_threshold = 0.7
        time_index = 0

        if time_index < 0 or time_index >= self.T:
            raise ValueError(f'time_index must be between 0 and {self.T - 1}, but received {time_index}.')

        N = self.traces.size(dim=0)
        if N == 0:
            raise ValueError('Traces tensor is empty, cannot evaluate cosine similarity.')

        one = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        zero = torch.tensor(-1.0, dtype=torch.float32, device=self.device)

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

                candidate_vec = self._evaluate_formula_on_traces(formula=candidate,batch_size=batch_size,time_index=time_index)
                candidate_vec = torch.where(candidate_vec, one, zero)
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
            sats = self._evaluate_formula_on_traces(formula=phi,batch_size=batch_size,time_index=time_index)
            vals = torch.where(sats, 
                               torch.tensor(1.0, dtype=torch.float32, device=self.device),
                               torch.tensor(0.0, dtype=torch.float32, device=self.device))  # (B,)
            F[i,:] = vals

        self.F = F



    # ----------- Kernel eval helper -----------
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


    def _compute_embedding_from_sats(self, phi_sats: torch.Tensor, move_to_cpu: bool) -> torch.Tensor:
        if self.F is None:
            raise ValueError("The Feature Matrix has not yet been built. Please do so using the build_F() method.")
        if self.traces is None:
            raise ValueError('Please sample traces before computing embeddings.')

        N = self.traces.size(dim=0)
        phi_vals = phi_sats.to(device=self.device, dtype=torch.float32)
        phi_centered = phi_vals - phi_vals.mean()
        F_centered = self.F - self.F.mean(dim=1, keepdim=True)
        emb = (F_centered @ phi_centered) / float(N)

        if move_to_cpu and self.device in ('cuda', 'mps'):
            emb = emb.cpu()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            else:
                torch.mps.empty_cache()
        return emb



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
        phi_sats = self._evaluate_formula_on_traces(formula=formula, batch_size=batch_size, time_index=time_index)
        return self._compute_embedding_from_sats(phi_sats, move_to_cpu=True)
    


    def compute_formula_embedding_no_move(self, formula: Formula, batch_size: int = 512, time_index: int = 0) -> torch.Tensor:
        """
        Method for computing the embedding of formula, from feature matrix F.
        - formula: the formula for which the embedding is to be calcualted.
        - batch size: (Default = 512) the size of the batches used during evaluation of the formula, adjustable for memory management.
        - time index: (Default = 0) the timepoint of the trace at which the formula is evaluated.
        Returns:
            - emb: Tensor (m), the embedding of formula, where m = len(self.anchor_formulas) the number of anchor formulae.
        """ 
        phi_sats = self._evaluate_formula_on_traces(formula=formula, batch_size=batch_size, time_index=time_index)
        return self._compute_embedding_from_sats(phi_sats, move_to_cpu=False)


    def compute_embedding_from_satisfaction(self, phi_sats: torch.Tensor, move_to_cpu: bool = False) -> torch.Tensor:
        """Compute the kernel embedding directly from a boolean satisfaction vector."""
        return self._compute_embedding_from_sats(phi_sats, move_to_cpu=move_to_cpu)




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