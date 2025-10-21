import itertools
import math
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
        self.F: torch.Tensor | None                 = None          # feature matrix (m, N), ±1, Tensor
        self.K: torch.Tensor | None                 = None          # kernel matrix (m, m), Tensor
        self.K0: torch.Tensor | None                = None          # cosine kernel matrix (m, m), Tensor



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



    def add_formulas(self, formulas: list):
        """
        Method for manually adding (a list of) formulae.
        """
        self.anchor_formulas.extend(formulas)



    def construct_anchor_formulas_kernel(self, m: int = 1024):
        """
        Method for constructing the set anchor formulae.
        - m: specifies the number of anchor formulae.
        """
        if self.T <= 0:
            raise ValueError("Trace length T must be positive to construct anchor formulas.")

        if self.AP <= 0:
            raise ValueError("Number of atomic propositions AP must be positive to construct anchor formulas.")

        if m < 2:
            raise ValueError("Parameter m must be at least 2 to determine a valid anchor set size.")

        log2_m = math.log2(m)
        if log2_m <= 0:
            raise ValueError("Parameter m must be greater than 1 to determine a valid anchor set size.")

        k = max(1, math.ceil((self.T * self.AP) / log2_m))

        anchor_times = list(range(0, self.T, k))
        num_anchor_times = len(anchor_times)

        # Each anchor time contributes 2^AP combinations, leading to |Chi| = 2^{AP * num_anchor_times}
        literal_cache = {
            (atom_idx, True): Atom(atom_idx)
            for atom_idx in range(self.AP)
        }
        literal_cache.update({
            (atom_idx, False): Not(literal_cache[(atom_idx, True)])
            for atom_idx in range(self.AP)
        })

        per_time_assignments = list(itertools.product((False, True), repeat=self.AP))

        Chi: list[Formula] = []
        for assignment in itertools.product(per_time_assignments, repeat=num_anchor_times):
            anchor_formula: Formula | None = None

            for time_idx, time_assignment in zip(anchor_times, assignment):
                conjunct: Formula | None = None
                for atom_idx, truth_value in enumerate(time_assignment):
                    literal = literal_cache[(atom_idx, truth_value)]
                    conjunct = literal if conjunct is None else And(conjunct, literal)

                assert conjunct is not None, "Conjunction over literals should never be empty."
                time_formula: Formula = conjunct
                for _ in range(time_idx):
                    time_formula = Next(time_formula)

                anchor_formula = time_formula if anchor_formula is None else And(anchor_formula, time_formula)

            assert anchor_formula is not None
            Chi.append(anchor_formula)

        self.add_formulas(Chi)



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
            # fill column i across batches
            j = 0
            while j < N:
                j1 = min(N, j + batch_size)
                batch = self.traces[j:j1]  # (B, AP, T)
                sats = eval_traces_batch(phi, batch)  # (B, T)
                vals = torch.where(sats[:, time_index], 
                                   torch.tensor(1.0, dtype=torch.float32, device=self.device),
                                   torch.tensor(0.0, dtype=torch.float32, device=self.device))  # (B,)
                F[i, j:j1] = vals
                j = j1
        
        self.F = F



    def build_K(self):
        """
        Method for building the kernel matrix, K, from feature matrix F. 
        Specifies self.K: 
        - K: Tensor (m, m) with values in [-N, N].
        """
        if self.F is None:
            raise ValueError("The Feature Matrix has not yet been built. Please do so using the build_F() method.")
        
        self.K = self.F @ self.F.T
        


    def normalize_K(self):
        """
        Method for normalizing the kernel matrix through cosine similarity [K0_ij = K_ij / sqrt(K_ii*K_jj)].
        Note that sqrt(K_ii*K_jj) = N, since K_ii = K_jj = N
        Specifies self.K0: 
        - K0: Tensor (m, m) with values in [-1, 1].
        """
        self.K0 = self.K / self.K[0,0].item()



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



    def compute_formula_embedding(self, formula: Formula, device: str, batch_size: int = 512, time_index: int = 0) -> torch.Tensor:
        """
        Method for computing the embedding of formula, from feature matrix F.
        - formula: the formula for which the embedding is to be calcualted.
        - batch size: (Default = 512) the size of the batches used during evaluation of the formula, adjustable for memory management.
        - time index: (Default = 0) the timepoint of the trace at which the formula is evaluated.
        Returns:
            - emb: Tensor (m), the embedding of formula, where m = len(self.anchor_formulas) the number of anchor formulae.
        """ 
        if self.F is None:
            raise ValueError("The Feature Matrix has not yet been built. Please do so using the build_F() method.")

        N = self.traces.size(dim=0)
        
        phi_sats = torch.empty(N, dtype=torch.float32, device=device)

        j = 0
        while j < N:
            j1 = min(N, j + batch_size)
            batch = self.traces[j:j1]  # (B, AP, T)
            batch_sats = eval_traces_batch(formula, batch)  # (B, T)
            vals = torch.where(batch_sats[:, time_index], 
                                torch.tensor(1.0, dtype=torch.float32, device=self.device),
                                torch.tensor(0.0, dtype=torch.float32, device=self.device))  # (B,)
            phi_sats[j:j1] = vals
            j = j1
            
        emb = self.F @ phi_sats # (m,)

        if self.device == 'cuda':
            emb = emb.cpu()
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            emb = emb.cpu() 
            torch.mps.empty_cache()
        
        return emb
    


    def compute_formula_embedding_no_move(self, formula: Formula, device: str, batch_size: int = 512, time_index: int = 0) -> torch.Tensor:
        """
        Method for computing the embedding of formula, from feature matrix F.
        - formula: the formula for which the embedding is to be calcualted.
        - batch size: (Default = 512) the size of the batches used during evaluation of the formula, adjustable for memory management.
        - time index: (Default = 0) the timepoint of the trace at which the formula is evaluated.
        Returns:
            - emb: Tensor (m), the embedding of formula, where m = len(self.anchor_formulas) the number of anchor formulae.
        """ 
        if self.F is None:
            raise ValueError("The Feature Matrix has not yet been built. Please do so using the build_F() method.")

        N = self.traces.size(dim=0)
        
        phi_sats = torch.empty(N, dtype=torch.float32, device=device)

        j = 0
        while j < N:
            j1 = min(N, j + batch_size)
            batch = self.traces[j:j1]  # (B, AP, T)
            batch_sats = eval_traces_batch(formula, batch)  # (B, T)
            vals = torch.where(batch_sats[:, time_index], 
                                torch.tensor(1.0, dtype=torch.float32, device=self.device),
                                torch.tensor(0.0, dtype=torch.float32, device=self.device))  # (B,)
            phi_sats[j:j1] = vals
            j = j1
            
        emb = self.F @ phi_sats # (m,)

        return emb
    


    def compute_formula_embedding_normalized(self, formula: Formula, device: str, batch_size: int = 512, time_index: int = 0) -> torch.Tensor:
        """
        Method for computing the embedding of formula, from feature matrix F.
        - formula: the formula for which the embedding is to be calcualted.
        - batch size: (Default = 512) the size of the batches used during evaluation of the formula, adjustable for memory management.
        - time index: (Default = 0) the timepoint of the trace at which the formula is evaluated.
        Returns:
            - emb: Tensor (m), the embedding of formula, where m = len(self.anchor_formulas) the number of anchor formulae.
        """ 
        if self.F is None:
            raise ValueError("The Feature Matrix has not yet been built. Please do so using the build_F() method.")

        N = self.traces.size(dim=0)
        
        phi_sats = torch.empty(N, dtype=torch.float32, device=device)

        j = 0
        while j < N:
            j1 = min(N, j + batch_size)
            batch = self.traces[j:j1]  # (B, AP, T)
            batch_sats = eval_traces_batch(formula, batch)  # (B, T)
            vals = torch.where(batch_sats[:, time_index], 
                                torch.tensor(1.0, dtype=torch.float32, device=self.device),
                                torch.tensor(0.0, dtype=torch.float32, device=self.device))  # (B,)
            phi_sats[j:j1] = vals
            j = j1
            
        emb = (self.F @ phi_sats) / N # (m,)

        if self.device == 'cuda':
            emb = emb.cpu()
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            emb = emb.cpu() 
            torch.mps.empty_cache()
        
        return emb