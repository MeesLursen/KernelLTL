from torch.utils.data import Dataset
import torch
from formula_class import Formula
from kernel_class import LTLKernel

class LTLDataset(Dataset):

    def __init__(self,
                 store_formula_str: bool = False,
                 store_satisfaction: bool = False,
                 satisfaction_batch_size: int = 512,
                 satisfaction_time_index: int = 0):
        self.store_formula_str = store_formula_str
        self.store_satisfaction = store_satisfaction
        self.satisfaction_batch_size = satisfaction_batch_size
        self.satisfaction_time_index = satisfaction_time_index

        self._reset_storage()


    def _reset_storage(self):
        self.formulas: list[Formula] = []
        self.embeddings: list[torch.Tensor] = []
        self.formula_strs: list[str] | None = [] if self.store_formula_str else None
        self.satisfactions: list[torch.Tensor] | None = [] if self.store_satisfaction else None


    def _append_entry(self,
                      formula: Formula,
                      embedding: torch.Tensor,
                      satisfaction: torch.Tensor | None):
        self.formulas.append(formula)
        self.embeddings.append(embedding.to(dtype=torch.float32, device='cpu'))

        if self.store_formula_str and self.formula_strs is not None:
            self.formula_strs.append(str(formula))

        if self.store_satisfaction:
            if satisfaction is None:
                raise ValueError("Satisfaction vector is required when store_satisfaction=True")
            if self.satisfactions is None:
                self.satisfactions = []
            self.satisfactions.append(satisfaction.to(dtype=torch.bool, device='cpu'))



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
        self._reset_storage()

        for phi in dataset_formulas:
            phi_sats = kernel._evaluate_formula_on_traces(
                formula=phi,
                batch_size=self.satisfaction_batch_size if self.store_satisfaction else batch_size,
                time_index=self.satisfaction_time_index
            )
            emb = kernel.compute_embedding_from_satisfaction(phi_sats, move_to_cpu=True)
            sats_to_store = phi_sats.clone().to('cpu') if self.store_satisfaction else None
            self._append_entry(phi, emb, sats_to_store)
    


    def construct_dataset_from_kernel_dedupe(self, kernel: LTLKernel, k: int, p_leaf: float, max_depth: int, batch_size: int = 512):
        """
        Method for constructing the dataset through the kernel, specifies self.formulas and self.embeddings.
        - kernel: the kernel we want to use for sampling formulae and computing their embeddings.
        - k: specifies the number of sampled formulae.
        - p_leaf: specifies the odds of each node being a leaf. Higher probability reduces average sampled formula complexity (bounded by max_depth).
        - max_depth: specifies the maximum formula complexity.
        - batch_size: (Default = 512) the size of the batches used during evaluation of the formulae, adjustable for memory management.
        """
        
        dataset_formulas = kernel.sample_dataset_formulas_kernel(k=k, p_leaf=p_leaf, max_depth=max_depth, force_tree=True)
        unique_formulas = list(dict.fromkeys(dataset_formulas))
        self._reset_storage()

        print(f'The deduplicated dataset contains {len(unique_formulas)} many formulae.')
        
        for phi in unique_formulas:
            phi_sats = kernel._evaluate_formula_on_traces(
                formula=phi,
                batch_size=self.satisfaction_batch_size if self.store_satisfaction else batch_size,
                time_index=self.satisfaction_time_index
            )
            emb = kernel.compute_embedding_from_satisfaction(phi_sats, move_to_cpu=True)
            sats_to_store = phi_sats.clone().to('cpu') if self.store_satisfaction else None
            self._append_entry(phi, emb, sats_to_store)
    


    def construct_dataset_from_list(self, input_formula_list: list[Formula], kernel: LTLKernel, batch_size: int = 512):
        """
        Method for constructing the dataset through the kernel, specifies self.formulas and self.embeddings.
        - kernel: the kernel we want to use for computing embeddings of the input formulae.
        - batch_size: (Default = 512) the size of the batches used during evaluation of the formulae, adjustable for memory management.
        """
        self._reset_storage()

        for phi in input_formula_list:
            phi_sats = kernel._evaluate_formula_on_traces(
                formula=phi,
                batch_size=self.satisfaction_batch_size if self.store_satisfaction else batch_size,
                time_index=self.satisfaction_time_index
            )
            emb = kernel.compute_embedding_from_satisfaction(phi_sats, move_to_cpu=True)
            sats_to_store = phi_sats.clone().to('cpu') if self.store_satisfaction else None
            self._append_entry(phi, emb, sats_to_store)


    
    def __len__(self):
        return len(self.formulas)
    


    def __getitem__(self, idx):
        print(f"DEBUG __getitem__ called with idx={idx}")
        if idx < 0 or idx >= len(self.formulas):
            raise IndexError(f"LTLDataset index out of range: {idx}")

        item = {
            "formula": self.formulas[idx],
            "embedding": self.embeddings[idx]
        }

        if self.store_formula_str and self.formula_strs is not None:
            item["formula_str"] = self.formula_strs[idx]

        if self.store_satisfaction and self.satisfactions is not None:
            item["satisfaction"] = self.satisfactions[idx]

        print(f"DEBUG __getitem__ returning keys={list(item.keys())}")
        return item