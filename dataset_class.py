from torch.utils.data import Dataset
import torch
from formula_class import Formula
from kernel_class import LTLKernel

class LTLDataset(Dataset):

    def __init__(self):
        self.formulas: list[Formula] = []
        self.embeddings: list[torch.Tensor] = []



    def construct_dataset_from_kernel(self, kernel: LTLKernel, k: int, p_leaf: float, max_depth: int, batch_size: int = 512):
        """
        Method for constructing the dataset through the kernel, specifies self.formulas and self.embeddings.
        - kernel: the kernel we want to use for sampling formulae and computing their embeddings.
        - k: specifies the number of sampled formulae.
        - p_leaf: specifies the odds of each node being a leaf. Higher probability reduces average sampled formula complexity (bounded by max_depth).
        - max_depth: specifies the maximum formula complexity.
        - batch_size: (Default = 512) the size of the batches used during evaluation of the formulae, adjustable for memory management.
        """
        
        dataset_formulas = kernel.sample_dataset_formulas_kernel(k=k, p_leaf=p_leaf, max_depth=max_depth, force_tree=True)
        self.formulas = dataset_formulas
        
        for phi in dataset_formulas:
            emb = kernel.compute_formula_embedding(phi, batch_size=batch_size, device=kernel.device)
            self.embeddings.append(emb)
    


    def construct_dataset_from_list(self, input_formula_list: list[Formula], kernel: LTLKernel, batch_size: int = 512):
        """
        Method for constructing the dataset through the kernel, specifies self.formulas and self.embeddings.
        - kernel: the kernel we want to use for computing embeddings of the input formulae.
        - batch_size: (Default = 512) the size of the batches used during evaluation of the formulae, adjustable for memory management.
        """
        self.formulas = input_formula_list

        for phi in input_formula_list:
            emb = kernel.compute_formula_embedding(phi, batch_size=batch_size, device=kernel.device)
            self.embeddings.append(emb)


    

    def __len__(self):
        return len(self.formulas)
    


    def __getitem__(self, idx):
        # returns CPU tensors/strings only
        return self.formulas[idx], self.embeddings[idx]