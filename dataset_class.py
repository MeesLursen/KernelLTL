from __future__ import annotations

from math import ceil
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
        self.metadata: dict[str, Any] = {"store_formula_str": self.store_formula_str,
                                         "store_satisfaction": self.store_satisfaction,
                                         "satisfaction_time_index": self.satisfaction_time_index,
                                         }

        self._reset_storage()



    def _reset_storage(self):
        self.formulas: list[Formula] = []
        self.embeddings: torch.Tensor | None = None
        self.formula_strs: list[str] | None = [] if self.store_formula_str else None
        self.satisfactions: torch.Tensor | None = None
        self.metadata: dict[str, any] = {"store_formula_str": self.store_formula_str,
                                         "store_satisfaction": self.store_satisfaction,
                                         "satisfaction_time_index": self.satisfaction_time_index,
                                         }



    def _append_entry(self,
                      formula: Formula,
                      embedding: torch.Tensor,
                      satisfaction: torch.Tensor | None):
        self.formulas.append(formula)
        emb = embedding.to(dtype=torch.float32, device='cpu').unsqueeze(0)
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = torch.cat([self.embeddings, emb], dim=0)

        if self.store_formula_str and self.formula_strs is not None:
            self.formula_strs.append(str(formula))

        if self.store_satisfaction:
            if satisfaction is None:
                raise ValueError("Satisfaction vector is required when store_satisfaction=True")
            sats = satisfaction.to(dtype=torch.bool, device='cpu').unsqueeze(0)
            if self.satisfactions is None:
                self.satisfactions = sats
            else:
                self.satisfactions = torch.cat([self.satisfactions, sats], dim=0)



    def _append_dataset(self: LTLDataset, other: LTLDataset) -> None:
        """Append entries from other into self (in-place)."""
        if self.store_formula_str != other.store_formula_str:
            raise ValueError("store_formula_str mismatch between base and other datasets")
        if self.store_satisfaction != other.store_satisfaction:
            raise ValueError("store_satisfaction mismatch between base and other datasets")
        for idx in range(len(other)):
            sats = None
            if self.store_satisfaction and other.satisfactions is not None:
                sats = other.satisfactions[idx]
            self._append_entry(other.formulas[idx], other.embeddings[idx], sats)



    def construct_dataset_from_kernel(self, kernel: LTLKernel, k: int, p_leaf_range: tuple[float,float], max_depth: int):
        """
        Method for constructing the dataset through the kernel, specifies self.formulas and self.embeddings.
        - kernel: the kernel we want to use for sampling formulae and computing their embeddings.
        - k: specifies the number of sampled formulae.
        - p_leaf_range: specifies the odds of each node being a leaf. Higher probability reduces average sampled formula complexity (bounded by max_depth).
        - max_depth: specifies the maximum formula complexity.
        - satisfaction_batch_size: configurable through LTLDataset to control evaluation batch sizes.
        """
        
        dataset_formulas = kernel.sample_dataset_formulas_kernel(k=k, p_leaf_range=p_leaf_range, max_depth=max_depth, force_tree=False)
        self._reset_storage()
        self.metadata = {
            "source": "kernel",
            "k": k,
            "p_leaf_range": p_leaf_range,
            "max_depth": max_depth,
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
    


    def construct_dataset_from_kernel_dedupe(self, kernel: LTLKernel, k: int, p_leaf_range: tuple[float,float], max_depth: int):
        """
        Method for constructing the dataset through the kernel, specifies self.formulas and self.embeddings.
        - kernel: the kernel we want to use for sampling formulae and computing their embeddings.
        - k: specifies the number of sampled formulae.
        - p_leaf_range: specifies the odds of each node being a leaf. Higher probability reduces average sampled formula complexity (bounded by max_depth).
        - max_depth: specifies the maximum formula complexity.
        - satisfaction_batch_size: configurable through LTLDataset to control evaluation batch sizes.
        """
        
        dataset_formulas = kernel.sample_dataset_formulas_kernel(k=k, p_leaf_range=p_leaf_range, max_depth=max_depth, force_tree=False)
        unique_formulas = list(dict.fromkeys(dataset_formulas))
        self._reset_storage()
        self.metadata.update({
            "source": "kernel_dedupe",
            "requested_k": k,
            "actual_k": len(unique_formulas),
            "p_leaf_range": p_leaf_range,
            "max_depth": max_depth,
            "kernel_T": kernel.T,
            "kernel_AP": kernel.AP,
            "kernel_seed": kernel.seed,
        })

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
        p_leaf_range: tuple[float,float],
        max_depth: int,
        min_depth: int | None = None,
        eval_ratio: float = 0.05,
        store_formula_str_train: bool = False,
        store_formula_str_eval: bool = True,
        store_satisfaction_train: bool = False,
        store_satisfaction_eval: bool = True,
        satisfaction_batch_size: int = 10240,
        satisfaction_time_index: int = 0,
        dedupe_eval: bool = True,
        exclude_formula_strs: set[str] | None = None,
    ) -> tuple[LTLDataset, LTLDataset]:
        """
        Sample k formulas and split into disjoint train/eval datasets using stratified sampling.
        
        Stratification ensures each depth level contributes proportionally to eval,
        preventing any single complexity level from being over-represented or depleted.
        
        Args:
            kernel: The kernel for sampling and computing embeddings.
            k: Total number of formulas to sample.
            p_leaf_range: Probability of leaf nodes during sampling.
            max_depth: Maximum formula depth during sampling.
            eval_ratio: Target fraction of formulas for eval (applied per depth stratum).
            store_formula_str_train: Whether to store formula strings in train dataset.
            store_formula_str_eval: Whether to store formula strings in eval dataset.
            store_satisfaction_train: Whether to store satisfactions in train dataset.
            store_satisfaction_eval: Whether to store satisfactions in eval dataset.
            satisfaction_batch_size: Batch size for satisfaction computation.
            satisfaction_time_index: Time index for satisfaction computation.
            dedupe_eval: If True, eval dataset contains only unique formulas.
        
        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        # Sample all formulas (uses kernel's RNG)
        print(f"Sampling {k} formulas with p_leaf_range={p_leaf_range}, max_depth={max_depth}...")
        all_formulas = kernel.sample_dataset_formulas_kernel(
            k=k, p_leaf_range=p_leaf_range, max_depth=max_depth, force_tree=False
        )

        if exclude_formula_strs:
            all_formulas = [phi for phi in all_formulas if str(phi) not in exclude_formula_strs]

        if min_depth is not None:
            all_formulas = [phi for phi in all_formulas if phi.depth() >= min_depth]
        
        if not all_formulas:
            raise ValueError("No formulas available after applying depth/exclusion filters. Consider reducing constraints or sampling more.")

        # Group formulas by their canonical string representation
        # Maps formula_str -> list of indices in all_formulas
        formula_groups: dict[str, list[int]] = defaultdict(list)
        for idx, phi in enumerate(all_formulas):
            formula_groups[str(phi)].append(idx)
        
        unique_formula_strs = list(formula_groups.keys())
        num_unique = len(unique_formula_strs)
        
        # Group unique formulas by depth for stratified sampling
        depth_groups: dict[int, list[str]] = defaultdict(list)
        for formula_str in unique_formula_strs:
            phi = all_formulas[formula_groups[formula_str][0]]
            depth = phi.depth()
            depth_groups[depth].append(formula_str)
        
        # Stratified selection: take eval_ratio from each depth group
        eval_formula_strs: set[str] = set()
        
        for depth in sorted(depth_groups.keys()):
            if depth == 0:
                continue
            formulas_at_depth = depth_groups[depth]
            n_at_depth = len(formulas_at_depth)
            
            # Shuffle formulas at this depth using kernel's RNG
            perm = torch.randperm(n_at_depth, generator=kernel.rng, device=kernel.device)
            shuffled = [formulas_at_depth[i] for i in perm.tolist()]
            
            # Take eval_ratio fraction, but at least 1 if there are any formulas
            n_eval_at_depth = max(1, int(n_at_depth * eval_ratio)) if n_at_depth > 0 else 0
            
            # Don't take more than half from any depth level to preserve training signal
            n_eval_at_depth = min(n_eval_at_depth, n_at_depth // 2)
            
            eval_formula_strs.update(shuffled[:n_eval_at_depth])
            
            print(f"  Depth {depth}: {n_at_depth} unique formulas, {n_eval_at_depth} -> eval")
        
        # Build index lists
        eval_indices: set[int] = set()
        for formula_str in eval_formula_strs:
            eval_indices.update(formula_groups[formula_str])
        
        n_available = len(all_formulas)
        train_indices = [i for i in range(n_available) if i not in eval_indices]
        eval_indices_list = sorted(eval_indices)
        
        print(f"Stratified disjoint split: {len(train_indices)} train, {len(eval_indices_list)} eval "
          f"({len(eval_indices_list) / n_available * 100:.1f}% eval)")
        print(f"Unique formulas: {num_unique} total, "
              f"{len(eval_formula_strs)} in eval, {num_unique - len(eval_formula_strs)} in train")
        
        # Create train dataset
        train_dataset = LTLDataset(
            store_formula_str=store_formula_str_train,
            store_satisfaction=store_satisfaction_train,
            satisfaction_batch_size=satisfaction_batch_size,
            satisfaction_time_index=satisfaction_time_index,
        )
        train_dataset._reset_storage()
                
        # Create eval dataset
        eval_dataset = LTLDataset(
            store_formula_str=store_formula_str_eval,
            store_satisfaction=store_satisfaction_eval,
            satisfaction_batch_size=satisfaction_batch_size,
            satisfaction_time_index=satisfaction_time_index,
        )
        eval_dataset._reset_storage()
        
        # Cache embeddings and satisfactions for unique formulas to avoid recomputation
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
        
        train_dataset.metadata.update({
            "source": "disjoint_split_train",
            "total_sampled_k": k,
            "train_count": len(train_dataset),
            "p_leaf_range": [p_leaf_range],
            "max_depth": max_depth,
            "kernel_T": kernel.T,
            "kernel_AP": kernel.AP,
            "kernel_seed": kernel.seed
        })

        # Clear caches - splits are disjoint, no reuse possible
        embedding_cache.clear()
        satisfaction_cache.clear()
        
        # Populate eval dataset
        print("Building eval dataset...")
        seen_in_eval: set[str] = set()
        for idx in eval_indices_list:
            phi = all_formulas[idx]
            phi_str = str(phi)
            
            # Skip duplicates in eval if dedupe_eval is enabled
            if dedupe_eval:
                if phi_str in seen_in_eval:
                    continue
                seen_in_eval.add(phi_str)
            
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
        
        
        eval_dataset.metadata.update({
            "source": "disjoint_split_eval",
            "total_sampled_k": k,
            "eval_count": len(eval_dataset),
            "p_leaf_range": [p_leaf_range],
            "max_depth": max_depth,
            "kernel_T": kernel.T,
            "kernel_AP": kernel.AP,
            "kernel_seed": kernel.seed,
            "dedupe_eval": dedupe_eval,
        })

        # Clear caches
        embedding_cache.clear()
        satisfaction_cache.clear()
       
        return train_dataset, eval_dataset


    
    def __len__(self):
        return len(self.formulas)
    


    def __getitem__(self, idx):
        item = {
            "formula": self.formulas[idx],
            "embedding": self.embeddings[idx] if self.embeddings is not None else None,
            "formula_id": idx,
        }

        if self.store_formula_str and self.formula_strs is not None:
            item["formula_str"] = self.formula_strs[idx]

        if self.store_satisfaction and self.satisfactions is not None:
            item["satisfaction"] = self.satisfactions[idx]

        return item

    
    
    def _delitem(self, idx):
        del self.formulas[idx]
        if self.embeddings is not None:
            self.embeddings = torch.cat([self.embeddings[:idx], self.embeddings[idx+1:]], dim=0)
        if self.store_formula_str and self.formula_strs is not None:
            del self.formula_strs[idx]
        if self.store_satisfaction and self.satisfactions is not None:
            self.satisfactions = torch.cat([self.satisfactions[:idx], self.satisfactions[idx+1:]], dim=0)


    # ----------- Persistence -----------
    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)

        num_examples = len(self.formulas)
        embedding_dim = self.embeddings.shape[1] if self.embeddings is not None and self.embeddings.ndim > 1 else 0

        metadata: dict[str, Any] = {
            "store_formula_str": self.store_formula_str,
            "store_satisfaction": self.store_satisfaction,
            "satisfaction_batch_size": self.satisfaction_batch_size,
            "satisfaction_time_index": self.satisfaction_time_index,
            "size": num_examples,
            "embedding_dim": embedding_dim,
            "has_satisfactions": self.store_satisfaction and self.satisfactions is not None and self.satisfactions.shape[0] == num_examples,
            "extra_metadata": self.metadata,
        }

        metadata_path = os.path.join(dirpath, "metadata.json")
        formulas_path = os.path.join(dirpath, "formulas.jsonl")
        embeddings_path = os.path.join(dirpath, "embeddings.pt")
        satisfactions_path = os.path.join(dirpath, "satisfactions.pt")

        with open(formulas_path, "w", encoding="utf-8") as fp:
            for formula in self.formulas:
                fp.write(str(formula) + "\n")

        if self.embeddings is not None and num_examples > 0:
            embeddings_tensor = self.embeddings.to(dtype=torch.float32, device="cpu")
        else:
            embeddings_tensor = torch.empty((0, embedding_dim), dtype=torch.float32)
        torch.save(embeddings_tensor, embeddings_path)

        if metadata["has_satisfactions"] and self.satisfactions is not None:
            sats_tensor = self.satisfactions.to(dtype=torch.bool, device="cpu")
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

        if os.path.exists(embeddings_path):
            dataset.embeddings = torch.load(embeddings_path, map_location="cpu")
        else:
            dataset.embeddings = None

        if dataset.store_formula_str and dataset.formula_strs is not None:
            dataset.formula_strs = [str(f) for f in formulas]

        if metadata.get("has_satisfactions") and os.path.exists(satisfactions_path):
            dataset.satisfactions = torch.load(satisfactions_path, map_location="cpu")
        else:
            dataset.satisfactions = None

        dataset.metadata = metadata.get("extra_metadata", {})
        return dataset


    @staticmethod
    def add_satisfactions_to_saved_dataset(
        dirpath: str,
        kernel: LTLKernel,
        satisfaction_batch_size: int | None = None,
        satisfaction_time_index: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        barrier_fn=None,
        prev_dirpath: str | None = None,
    ) -> None:
        """
        Compute satisfactions from formulas in a saved dataset directory and write
        them to `satisfactions.pt` in that same directory. DDP-compatible: splits work by rank.

        Each rank writes its chunk to `satisfactions.part{rank}.pt`. Rank 0 aggregates after all finish.

        Args:
            dirpath: Path to a saved LTLDataset directory.
            kernel: Kernel used to evaluate formula satisfactions on traces.
            satisfaction_batch_size: Optional override for evaluation batch size.
            satisfaction_time_index: Optional override for evaluation time index.
            rank: DDP rank (default 0 for single process).
            world_size: DDP world size (default 1 for single process).
            barrier_fn: Optional callable for DDP barrier (e.g., torch.distributed.barrier).
            prev_dirpath: Path to the previous stage LTLDataset directory.
        """
        metadata_path = os.path.join(dirpath, "metadata.json")
        formulas_path = os.path.join(dirpath, "formulas.jsonl")
        satisfactions_path = os.path.join(dirpath, "satisfactions.pt")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Dataset metadata not found in {dirpath}")
        if not os.path.exists(formulas_path):
            raise FileNotFoundError(f"Dataset formulas not found in {dirpath}")

        with open(metadata_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)

        batch_size = (
            satisfaction_batch_size
            if satisfaction_batch_size is not None
            else metadata.get("satisfaction_batch_size", 512)
        )
        time_index = (
            satisfaction_time_index
            if satisfaction_time_index is not None
            else metadata.get("satisfaction_time_index", 0)
        )


        formulas: list[Formula] = []
        with open(formulas_path, "r", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if text:
                    formulas.append(str_to_formula(text))

        prev_sats = None
        prev_len = 0
        if prev_dirpath is not None:
            prev_sats_path = os.path.join(prev_dirpath, "satisfactions.pt")
            if os.path.exists(prev_sats_path):
                prev_sats = torch.load(prev_sats_path, map_location="cpu")
                prev_len = prev_sats.shape[0]
            else:
                raise FileNotFoundError(f"Previous satisfactions.pt not found in {prev_dirpath}")

        # Only compute new satisfactions for formulas after prev_len
        total = len(formulas)
        new_start = prev_len
        new_total = total - prev_len
        chunk_size = (new_total + world_size - 1) // world_size
        start = new_start + rank * chunk_size
        end = min(new_start + (rank + 1) * chunk_size, total)
        formulas_chunk = formulas[start:end] if start < end else []

        satisfactions: list[torch.Tensor] = []
        for phi in formulas_chunk:
            phi_sats = kernel._evaluate_formula_on_traces(
                formula=phi,
                batch_size=batch_size,
                time_index=time_index,
            )
            satisfactions.append(phi_sats.to(dtype=torch.bool, device="cpu"))

        # Save partial chunk
        part_path = os.path.join(dirpath, f"satisfactions.part{rank}.pt")
        if satisfactions:
            sats_tensor = torch.stack(satisfactions, dim=0).to(dtype=torch.bool, device="cpu")
        else:
            sats_tensor = torch.empty((0,), dtype=torch.bool)
        torch.save(sats_tensor, part_path)
        print(f'rank {rank} finished and saved tensor.')

        # Barrier for all ranks to finish
        if barrier_fn is not None:
            barrier_fn()

        # Only rank 0 aggregates
        if rank == 0:
            # Wait for all part files
            import time
            for r in range(world_size):
                wait_path = os.path.join(dirpath, f"satisfactions.part{r}.pt")
                while not os.path.exists(wait_path):
                    time.sleep(1)
            # Concatenate all parts for new formulas
            all_parts = [torch.load(os.path.join(dirpath, f"satisfactions.part{r}.pt"), map_location="cpu") for r in range(world_size)]
            if all_parts:
                new_sats_tensor = torch.cat(all_parts, dim=0)
            else:
                new_sats_tensor = torch.empty((0,), dtype=torch.bool)
                print("No new sats added. This is an indicator something is wrong.")
            # Concatenate previous and new satisfactions in order
            if prev_sats is not None:
                sats_tensor = torch.cat([prev_sats, new_sats_tensor], dim=0)
            else:
                sats_tensor = new_sats_tensor
            torch.save(sats_tensor, satisfactions_path)
            # Clean up part files
            for r in range(world_size):
                os.remove(os.path.join(dirpath, f"satisfactions.part{r}.pt"))

            metadata["store_satisfaction"] = True
            metadata["has_satisfactions"] = True
            metadata["satisfaction_batch_size"] = batch_size
            metadata["satisfaction_time_index"] = time_index

            with open(metadata_path, "w", encoding="utf-8") as fp:
                json.dump(metadata, fp, indent=2)