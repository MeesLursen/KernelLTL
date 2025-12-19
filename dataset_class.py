from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import Dataset

from formula_class import Formula
from formula_utils import str_to_formula
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
        self.metadata: dict[str, Any] = {}

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



    def construct_dataset_from_kernel(self, kernel: LTLKernel, k: int, p_leaf: float, max_depth: int):
        """
        Method for constructing the dataset through the kernel, specifies self.formulas and self.embeddings.
        - kernel: the kernel we want to use for sampling formulae and computing their embeddings.
        - k: specifies the number of sampled formulae.
        - p_leaf: specifies the odds of each node being a leaf. Higher probability reduces average sampled formula complexity (bounded by max_depth).
        - max_depth: specifies the maximum formula complexity.
        - satisfaction_batch_size: configurable through LTLDataset to control evaluation batch sizes.
        """
        
        dataset_formulas = kernel.sample_dataset_formulas_kernel(k=k, p_leaf=p_leaf, max_depth=max_depth, force_tree=True)
        self._reset_storage()
        self.metadata = {
            "source": "kernel",
            "k": k,
            "p_leaf": p_leaf,
            "max_depth": max_depth,
            "batch_size": self.satisfaction_batch_size,
            "kernel_T": kernel.T,
            "kernel_AP": kernel.AP,
            "kernel_seed": kernel.seed,
        }

        for phi in dataset_formulas:
            phi_sats = kernel._evaluate_formula_on_traces(
                formula=phi,
                batch_size=self.satisfaction_batch_size,
                time_index=self.satisfaction_time_index
            )
            emb = kernel.compute_embedding_from_satisfaction(phi_sats, move_to_cpu=True)
            sats_to_store = phi_sats.clone().to('cpu') if self.store_satisfaction else None
            self._append_entry(phi, emb, sats_to_store)
    


    def construct_dataset_from_kernel_dedupe(self, kernel: LTLKernel, k: int, p_leaf: float, max_depth: int):
        """
        Method for constructing the dataset through the kernel, specifies self.formulas and self.embeddings.
        - kernel: the kernel we want to use for sampling formulae and computing their embeddings.
        - k: specifies the number of sampled formulae.
        - p_leaf: specifies the odds of each node being a leaf. Higher probability reduces average sampled formula complexity (bounded by max_depth).
        - max_depth: specifies the maximum formula complexity.
        - satisfaction_batch_size: configurable through LTLDataset to control evaluation batch sizes.
        """
        
        dataset_formulas = kernel.sample_dataset_formulas_kernel(k=k, p_leaf=p_leaf, max_depth=max_depth, force_tree=True)
        unique_formulas = list(dict.fromkeys(dataset_formulas))
        self._reset_storage()
        self.metadata = {
            "source": "kernel_dedupe",
            "requested_k": k,
            "actual_k": len(unique_formulas),
            "p_leaf": p_leaf,
            "max_depth": max_depth,
            "batch_size": self.satisfaction_batch_size,
            "kernel_T": kernel.T,
            "kernel_AP": kernel.AP,
            "kernel_seed": kernel.seed,
        }

        print(f'The deduplicated dataset contains {len(unique_formulas)} many formulae.')
        
        for phi in unique_formulas:
            phi_sats = kernel._evaluate_formula_on_traces(
                formula=phi,
                batch_size=self.satisfaction_batch_size,
                time_index=self.satisfaction_time_index
            )
            emb = kernel.compute_embedding_from_satisfaction(phi_sats, move_to_cpu=True)
            sats_to_store = phi_sats.clone().to('cpu') if self.store_satisfaction else None
            self._append_entry(phi, emb, sats_to_store)
    


    def construct_dataset_from_list(self, input_formula_list: list[Formula], kernel: LTLKernel):
        """
        Method for constructing the dataset through the kernel, specifies self.formulas and self.embeddings.
        - kernel: the kernel we want to use for computing embeddings of the input formulae.
        - satisfaction_batch_size: configurable through LTLDataset to control evaluation batch sizes.
        """
        self._reset_storage()
        self.metadata = {
            "source": "list",
            "count": len(input_formula_list),
            "batch_size": self.satisfaction_batch_size,
            "kernel_T": kernel.T,
            "kernel_AP": kernel.AP,
            "kernel_seed": kernel.seed,
        }

        for phi in input_formula_list:
            phi_sats = kernel._evaluate_formula_on_traces(
                formula=phi,
                batch_size=self.satisfaction_batch_size,
                time_index=self.satisfaction_time_index
            )
            emb = kernel.compute_embedding_from_satisfaction(phi_sats, move_to_cpu=True)
            sats_to_store = phi_sats.clone().to('cpu') if self.store_satisfaction else None
            self._append_entry(phi, emb, sats_to_store)


    @staticmethod
    def construct_disjoint_datasets(
        kernel: LTLKernel,
        k: int,
        p_leaf: float,
        max_depth: int,
        eval_ratio: float = 0.05,
        store_formula_str_train: bool = False,
        store_formula_str_eval: bool = True,
        store_satisfaction_train: bool = False,
        store_satisfaction_eval: bool = True,
        satisfaction_batch_size: int = 512,
        satisfaction_time_index: int = 0,
    ) -> tuple[LTLDataset, LTLDataset]:
        """
        Construct disjoint train and eval datasets from a single sample of formulas.
        
        This method samples k formulas, groups duplicates together, and then randomly
        moves entire groups to the eval set until the eval set contains at least
        `eval_ratio` fraction of all sampled formulas. This ensures that:
        1. Train and eval sets have no overlapping formulas
        2. All copies of a formula end up in the same split
        
        Args:
            kernel: The kernel to use for sampling and computing embeddings.
                    The kernel's RNG is used for both sampling and splitting.
            k: Total number of formulas to sample.
            p_leaf: Probability of a node being a leaf during sampling.
            max_depth: Maximum tree depth for sampled formulas.
            eval_ratio: Target fraction of formulas for evaluation (default 0.05 = 5%).
            store_formula_str_train: Whether to store formula strings in train dataset.
            store_formula_str_eval: Whether to store formula strings in eval dataset.
            store_satisfaction_train: Whether to store satisfaction tensors in train dataset.
            store_satisfaction_eval: Whether to store satisfaction tensors in eval dataset.
            satisfaction_batch_size: Batch size for computing satisfactions.
            satisfaction_time_index: Time index for satisfaction computation.
        
        Returns:
            A tuple (train_dataset, eval_dataset) with disjoint formulas.
        """
        # Sample all formulas (uses kernel's RNG)
        all_formulas = kernel.sample_dataset_formulas_kernel(
            k=k, p_leaf=p_leaf, max_depth=max_depth, force_tree=True
        )
        
        # Group formulas by their canonical string representation
        # Maps formula_str -> list of indices in all_formulas
        formula_groups: dict[str, list[int]] = defaultdict(list)
        for idx, phi in enumerate(all_formulas):
            formula_groups[str(phi)].append(idx)
        
        unique_formula_strs = list(formula_groups.keys())
        num_unique = len(unique_formula_strs)
        
        # Shuffle the unique formulas for random splitting using kernel's RNG
        perm = torch.randperm(num_unique, generator=kernel.rng, device=kernel.device)
        unique_formula_strs = [unique_formula_strs[i] for i in perm.tolist()]
        
        # Select groups for eval until we reach the target ratio
        target_eval_count = int(k * eval_ratio)
        eval_indices: set[int] = set()
        eval_formula_strs: set[str] = set()
        
        for formula_str in unique_formula_strs:
            if len(eval_indices) >= target_eval_count:
                break
            group_indices = formula_groups[formula_str]
            eval_indices.update(group_indices)
            eval_formula_strs.add(formula_str)
        
        # Remaining indices go to train
        train_indices = [i for i in range(k) if i not in eval_indices]
        eval_indices_list = sorted(eval_indices)
        
        print(f"Disjoint split: {len(train_indices)} train, {len(eval_indices_list)} eval "
              f"({len(eval_indices_list) / k * 100:.1f}% eval)")
        print(f"Unique formulas: {len(unique_formula_strs)} total, "
              f"{len(eval_formula_strs)} in eval, {len(unique_formula_strs) - len(eval_formula_strs)} in train")
        
        # Create train dataset
        train_dataset = LTLDataset(
            store_formula_str=store_formula_str_train,
            store_satisfaction=store_satisfaction_train,
            satisfaction_batch_size=satisfaction_batch_size,
            satisfaction_time_index=satisfaction_time_index,
        )
        train_dataset._reset_storage()
        train_dataset.metadata = {
            "source": "disjoint_split_train",
            "total_sampled_k": k,
            "train_count": len(train_indices),
            "eval_count": len(eval_indices_list),
            "eval_ratio_target": eval_ratio,
            "eval_ratio_actual": len(eval_indices_list) / k,
            "unique_formulas_total": num_unique,
            "unique_formulas_train": num_unique - len(eval_formula_strs),
            "unique_formulas_eval": len(eval_formula_strs),
            "p_leaf": p_leaf,
            "max_depth": max_depth,
            "kernel_T": kernel.T,
            "kernel_AP": kernel.AP,
            "kernel_seed": kernel.seed,
        }
        
        # Create eval dataset
        eval_dataset = LTLDataset(
            store_formula_str=store_formula_str_eval,
            store_satisfaction=store_satisfaction_eval,
            satisfaction_batch_size=satisfaction_batch_size,
            satisfaction_time_index=satisfaction_time_index,
        )
        eval_dataset._reset_storage()
        eval_dataset.metadata = {
            "source": "disjoint_split_eval",
            "total_sampled_k": k,
            "train_count": len(train_indices),
            "eval_count": len(eval_indices_list),
            "eval_ratio_target": eval_ratio,
            "eval_ratio_actual": len(eval_indices_list) / k,
            "unique_formulas_total": num_unique,
            "unique_formulas_train": num_unique - len(eval_formula_strs),
            "unique_formulas_eval": len(eval_formula_strs),
            "p_leaf": p_leaf,
            "max_depth": max_depth,
            "kernel_T": kernel.T,
            "kernel_AP": kernel.AP,
            "kernel_seed": kernel.seed,
        }
        
        # Cache embeddings and satisfactions for unique formulas to avoid recomputation
        # since we may have duplicates
        embedding_cache: dict[str, torch.Tensor] = {}
        satisfaction_cache: dict[str, torch.Tensor] = {}
        
        # Populate train dataset
        print("Building train dataset...")
        for idx in train_indices:
            phi = all_formulas[idx]
            phi_str = str(phi)
            
            if phi_str not in embedding_cache:
                phi_sats = kernel._evaluate_formula_on_traces(
                    formula=phi,
                    batch_size=satisfaction_batch_size,
                    time_index=satisfaction_time_index,
                )
                embedding_cache[phi_str] = kernel.compute_embedding_from_satisfaction(phi_sats, move_to_cpu=True)
                if store_satisfaction_train:
                    satisfaction_cache[phi_str] = phi_sats.clone().to('cpu')
            
            emb = embedding_cache[phi_str]
            sats_to_store = satisfaction_cache.get(phi_str) if store_satisfaction_train else None
            train_dataset._append_entry(phi, emb, sats_to_store)
        
        # Populate eval dataset
        print("Building eval dataset...")
        for idx in eval_indices_list:
            phi = all_formulas[idx]
            phi_str = str(phi)
            
            if phi_str not in embedding_cache:
                phi_sats = kernel._evaluate_formula_on_traces(
                    formula=phi,
                    batch_size=satisfaction_batch_size,
                    time_index=satisfaction_time_index,
                )
                embedding_cache[phi_str] = kernel.compute_embedding_from_satisfaction(phi_sats, move_to_cpu=True)
                if store_satisfaction_eval:
                    satisfaction_cache[phi_str] = phi_sats.clone().to('cpu')
            
            emb = embedding_cache[phi_str]
            sats_to_store = satisfaction_cache.get(phi_str) if store_satisfaction_eval else None
            eval_dataset._append_entry(phi, emb, sats_to_store)
        
        return train_dataset, eval_dataset


    
    def __len__(self):
        return len(self.formulas)
    


    def __getitem__(self, idx):
        item = {
            "formula": self.formulas[idx],
            "embedding": self.embeddings[idx]
        }

        if self.store_formula_str and self.formula_strs is not None:
            item["formula_str"] = self.formula_strs[idx]

        if self.store_satisfaction and self.satisfactions is not None:
            item["satisfaction"] = self.satisfactions[idx]

        return item


    # ----------- Persistence -----------
    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)

        num_examples = len(self.formulas)
        embedding_dim = self.embeddings[0].numel() if self.embeddings else 0

        metadata: dict[str, Any] = {
            "store_formula_str": self.store_formula_str,
            "store_satisfaction": self.store_satisfaction,
            "satisfaction_batch_size": self.satisfaction_batch_size,
            "satisfaction_time_index": self.satisfaction_time_index,
            "size": num_examples,
            "embedding_dim": embedding_dim,
            "has_satisfactions": self.store_satisfaction and self.satisfactions is not None and len(self.satisfactions) == num_examples,
            "extra_metadata": self.metadata,
        }

        metadata_path = os.path.join(dirpath, "metadata.json")
        formulas_path = os.path.join(dirpath, "formulas.jsonl")
        embeddings_path = os.path.join(dirpath, "embeddings.pt")
        satisfactions_path = os.path.join(dirpath, "satisfactions.pt")

        with open(formulas_path, "w", encoding="utf-8") as fp:
            for formula in self.formulas:
                fp.write(str(formula) + "\n")

        if num_examples > 0:
            embeddings_tensor = torch.stack(self.embeddings, dim=0).to(dtype=torch.float32, device="cpu")
        else:
            embeddings_tensor = torch.empty((0, embedding_dim), dtype=torch.float32)
        torch.save(embeddings_tensor, embeddings_path)

        if metadata["has_satisfactions"] and self.satisfactions is not None:
            sats_tensor = torch.stack(self.satisfactions, dim=0).to(dtype=torch.bool, device="cpu")
            torch.save(sats_tensor, satisfactions_path)

        with open(metadata_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)


    @classmethod
    def load(cls, dirpath: str) -> "LTLDataset":
        metadata_path = os.path.join(dirpath, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Dataset metadata not found in {dirpath}")

        with open(metadata_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)

        dataset = cls(
            store_formula_str=metadata.get("store_formula_str", False),
            store_satisfaction=metadata.get("store_satisfaction", False),
            satisfaction_batch_size=metadata.get("satisfaction_batch_size", 512),
            satisfaction_time_index=metadata.get("satisfaction_time_index", 0)
        )

        formulas_path = os.path.join(dirpath, "formulas.jsonl")
        embeddings_path = os.path.join(dirpath, "embeddings.pt")
        satisfactions_path = os.path.join(dirpath, "satisfactions.pt")

        formulas: list[Formula] = []
        if os.path.exists(formulas_path):
            with open(formulas_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    text = line.strip()
                    if text:
                        formulas.append(str_to_formula(text))
        dataset.formulas = formulas

        embeddings_tensor = torch.load(embeddings_path, map_location="cpu")
        dataset.embeddings = [embeddings_tensor[i].clone().detach() for i in range(embeddings_tensor.size(0))]

        if dataset.store_formula_str and dataset.formula_strs is not None:
            dataset.formula_strs = [str(f) for f in formulas]

        if metadata.get("has_satisfactions") and os.path.exists(satisfactions_path):
            sats_tensor = torch.load(satisfactions_path, map_location="cpu")
            dataset.satisfactions = [sats_tensor[i].clone().detach() for i in range(sats_tensor.size(0))]

        dataset.metadata = metadata.get("extra_metadata", {})
        return dataset