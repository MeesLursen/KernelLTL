from typing import List, Dict, Optional
import numpy as np
#import tensorflow as tf

class LTLKernel:
    def __init__(self, L: int, n_ap: int):
        """
        Kernel for LTL formulas based on sampled traces.

        Args:
            L (int): Maximum trace length.
            n_ap (int): Number of atomic propositions.
        """
        self.L = L
        self.n_ap = n_ap

        self.traces = None       # (N, L, n_ap), bool
        self.formulas = []       # list of parsed formula objects or strings
        self.F = None            # feature matrix (m, N), ±1 as tf.Tensor
        self.K = None            # kernel matrix (m, m), tf.Tensor

    
    
    # ----------- Sampling -----------
    def sample_traces_1(
        self,
        N: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample N traces uniformly at random.
        Each trace is shape (n_ap, L), with values in {0,1}.
        """
        rng = np.random.default_rng(seed)
        self.traces = rng.integers(0, 2, size=(N, self.n_ap, self.L), dtype=np.uint8).astype(bool)

        return self.traces
    
    
    def sample_traces_2(
        self,
        N: int,
        props: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> List[List[Dict[str, bool]]]:
        """
        Return N traces, each a List[Dict[str, bool]] of length `length`.
        Each timestep is a dict mapping each proposition name in `props` to a bool.
        Sampling is uniform over all valuations (each proposition is an independent fair bit).
        """
        
        if props == None:
            props = [f"p{i}" for i in range(self.n_ap)]        
        rng = np.random.default_rng(seed)
        ls_traces = []

        for _ in range(N):
            trace = []

            for _ in range(self.L):
                valuation = {a: bool(rng.bit_generator) for a in props}
                trace.append(valuation)

            ls_traces.append(trace)

        self.trace = trace
        return self.trace
        

'''
    # ----------- Formula Management -----------
    def add_formulas(self, formulas):
        """
        Add formulas (strings or ASTs).
        In future, parsing can be done here.
        """
        self.formulas.extend(formulas)

    # ----------- Evaluation -----------
    def eval_formula_on_trace(self, phi, trace) -> int:
        """
        Evaluate formula phi on a single trace.
        Must return ±1. Placeholder for now.
        """
        raise NotImplementedError("Formula evaluation not implemented yet.")

    def build_feature_matrix(self):
        """
        Build F (m, N) where F[i,j] = ±1 = s(phi_i, trace_j).
        Stored as a TensorFlow tensor for GPU support.
        """
        if self.traces is None or not self.formulas:
            raise ValueError("Need both traces and formulas before building feature matrix.")

        m = len(self.formulas)
        N = self.traces.shape[0]

        # placeholder with +1 everywhere (to be replaced by evaluator)
        F = np.ones((m, N), dtype=np.int8)

        # TODO: loop over formulas and traces to fill F
        # for i, phi in enumerate(self.formulas):
        #     for j, trace in enumerate(self.traces):
        #         F[i, j] = self.eval_formula_on_trace(phi, trace)

        self.F = tf.convert_to_tensor(F, dtype=tf.float32)
        return self.F

    # ----------- Kernel Computation -----------
    def compute_kernel(self, normalize: bool = False):
        """
        Compute kernel K = F @ F.T (optionally normalized).
        """
        if self.F is None:
            raise ValueError("Feature matrix F not built yet.")

        K = tf.matmul(self.F, self.F, transpose_b=True)

        if normalize:
            diag = tf.linalg.diag_part(K)
            norm = tf.sqrt(tf.expand_dims(diag, 1) * tf.expand_dims(diag, 0))
            K = tf.math.divide_no_nan(K, norm)

        self.K = K
        return self.K

    # ----------- Diagnostics -----------
    def diagnostics(self):
        """
        Print simple diagnostics about traces and feature matrix.
        """
        print(f"L={self.L}, n_ap={self.n_ap}")
        if self.traces is not None:
            print(f"Traces: {self.traces.shape}")
        if self.F is not None:
            print(f"Feature matrix: {self.F.shape}")
        if self.K is not None:
            print(f"Kernel matrix: {self.K.shape}")
'''