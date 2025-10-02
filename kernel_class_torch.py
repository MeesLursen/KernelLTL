import torch
from formula_class_torch import sample_traces_torch, sample_formulas_torch, eval_traces_batch_torch

class LTLKernel_torch:
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
                                     else
                                     torch.Generator(device=self.device))
        
        self.formulas: list                           = []            # list of parsed formula objects or strings
        self.traces: torch.Tensor | None              = None          # (N, AP, T), bool, Tensor
        self.F: torch.Tensor | None                   = None          # feature matrix (m, N), ±1, Tensor
        self.K: torch.Tensor | None                   = None          # kernel matrix (m, m), Tensor
        self.K0: torch.Tensor | None                  = None          # cosine kernel matrix (m, m), Tensor



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
        self.traces = sample_traces_torch(N,
                                          n_ap=self.AP,
                                          trace_length=self.T,
                                          rng=self.rng,
                                          device=self.device)



    def add_formulas(self, formulas: list):
        """
        Method for manually adding (a list of) formulae.
        """
        self.formulas.extend(formulas)



    def sample_formulas_kernel(self, 
                               m: int,
                               p_leaf: float    = 0.5,
                               max_depth: int   = 6,
                               force_tree: bool = True
                               ):
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
        sample = sample_formulas_torch(n_formula=m,
                                       p_leaf=p_leaf,
                                       max_depth=max_depth,
                                       n_ap=self.AP,
                                       force_tree=force_tree,
                                       rng=self.rng,
                                       device=self.device)

        self.add_formulas(sample)



    # ----------- Evaluation -----------
    def build_F(self, batch_size: int = 512, time_index: int = 0) -> torch.Tensor:
        """
        Method for building the feature matrix F from the sampled formulae and traces.
        - formulas: list of formulae length m.
        - all_traces: Tensor shape (N, AP, T), dtype=bool.
        Specifies self.F: 
        - F: Tensor of shape (m, N) with ±1 values, dtype=int8.
        """
        if self.traces is None and self.formulas is []:
            raise ValueError('Please first sample traces and formulas, using the sample_traces(N) and sample_formulas() method respectively.')

        if not(self.traces is None) and self.formulas is []:
            raise ValueError('You have not yet sampled formulas. Please do so using the sample_formulas() method.')
        
        if self.traces is None and not(self.formulas is []):
            raise ValueError('You have not yet sampled traces. Please do so using the sample_traces() method.')
        

        N = self.traces.shape[0]
        m = len(self.formulas)
        F = torch.empty((m, N), dtype=torch.int8)
        for i, phi in enumerate(self.formulas):
            # fill column i across batches
            j = 0
            while j < N:
                j1 = min(N, j + batch_size)
                batch = self.traces[j:j1]  # (B, AP, T)
                print(batch.type())
                sats = eval_traces_batch_torch(phi, batch)  # (B, T)
                print(sats.type())
                vals = torch.where(sats[:, time_index], 1, -1).to(torch.int8)  # (B,)
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
        
        F32 = self.F.to(torch.int32)
        self.K = F32 @ F32.T
        


    def normalize_K(self):
        """
        Method for normalizing the kernel matrix through cosine similarity [K0_ij = K_ij / sqrt(K_ii*K_jj)].
        Specifies self.K0: 
        - K0: Tensor (m, m) with values in [-1, 1].
        """
        raise NotImplementedError