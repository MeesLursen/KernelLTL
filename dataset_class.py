from torch.utils.data import Dataset
import torch
from formula_class import Formula
from kernel_class import LTLKernel

class LTLDataset(Dataset):

    def __init__(self, items: list[tuple[Formula, torch.Tensor]]):
        self.formulas = []
        self.embeddings = []
        self.device = 'cpu'

    def construct_dataset_from_kernel(self, kernel: LTLKernel, k: int, p_leaf: float, max_depth: int, batch_size: int = 512) -> list[tuple[Formula, torch.Tensor]]:
        """
        Method for constructing the input dataset.
        - input_formula_list: an list of formulae for which we wish to calculate the embedding (w.r.t. set of anchor formlae self.anchor_formlas).
        - batch_size: (Default = 512) the size of the batches used during evaluation of the formula, adjustable for memory management.
        Returns:
            - dataset: a list of tuple (formula, embedding)
        """
        
        dataset_formulas = kernel.sample_dataset_formulas_kernel(k=k, p_leaf=p_leaf, max_depth=max_depth, force_tree=True)
        ls = []

        for phi in dataset_formulas:
            emb = kernel.compute_formula_embedding(phi, batch_size=batch_size, device=self.device) # (m,) # TODO: check if this works, see compute_formula_embedding in kernel_class.py!!!
            ls.append((phi, emb))
        
        return ls
        

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        # returns CPU tensors/strings only
        return self.formulas[idx], self.embeddings[idx]