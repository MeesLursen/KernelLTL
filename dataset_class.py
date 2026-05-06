from __future__ import annotations

from math import ceil
import json
import os
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import Dataset

from formula_class import Formula
from formula_utils import (
    str_to_formula,
    list_semantically_equivalent_transformations,
    list_negation_insertions,
)
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
    def construct_dataset_from_kernel_excluding(
        kernel: LTLKernel,
        k: int,
        p_leaf_range: tuple[float, float],
        max_depth: int,
        min_depth: int | None = None,
        depth_targets: dict[int, int] | None = None,
        exclude_formula_strs: set[str] | None = None,
        dedupe: bool = True,
        store_formula_str: bool = True,
        store_satisfaction: bool = True,
        satisfaction_batch_size: int = 10240,
        satisfaction_time_index: int = 0,
        max_sampling_attempts: int = 100,
    ) -> LTLDataset:
        """
        Sample a dataset from kernel while excluding formulas by canonical string.

        If depth_targets is provided, enforce exact per-depth counts by repeatedly
        sampling and only accepting formulas from depths that are still missing.
        """

        if k <= 0:
            raise ValueError("k must be > 0")
        if max_sampling_attempts <= 0:
            raise ValueError("max_sampling_attempts must be > 0")
        if depth_targets is not None and not dedupe:
            raise ValueError("depth_targets requires dedupe=True")

        validated_depth_targets: dict[int, int] | None = None
        if depth_targets is not None:
            if len(depth_targets) == 0:
                raise ValueError("depth_targets must not be empty when provided")
            validated_depth_targets = {}
            for depth, target_count in depth_targets.items():
                depth_i = int(depth)
                target_i = int(target_count)
                if target_i <= 0:
                    raise ValueError(f"depth_targets[{depth_i}] must be > 0")
                if min_depth is not None and depth_i < min_depth:
                    raise ValueError(
                        f"depth_targets contains depth {depth_i} smaller than min_depth={min_depth}"
                    )
                if depth_i > max_depth:
                    raise ValueError(
                        f"depth_targets contains depth {depth_i} larger than max_depth={max_depth}"
                    )
                validated_depth_targets[depth_i] = target_i

            target_total = sum(validated_depth_targets.values())
            if target_total != k:
                raise ValueError(
                    f"k ({k}) must equal sum(depth_targets) ({target_total}) when depth_targets is set"
                )

        exclude = set(exclude_formula_strs) if exclude_formula_strs is not None else set()
        selected_formulas: list[Formula] = []
        selected_strs: set[str] = set() if dedupe else set()
        rejected_excluded = 0
        rejected_depth = 0
        rejected_duplicate = 0
        rejected_filled_depth = 0

        per_depth_selected: dict[int, list[Formula]] = {}
        per_depth_counts: dict[int, int] = {}
        if validated_depth_targets is not None:
            per_depth_selected = {depth: [] for depth in validated_depth_targets}
            per_depth_counts = {depth: 0 for depth in validated_depth_targets}

        attempts = 0
        while attempts < max_sampling_attempts:
            if validated_depth_targets is None:
                if len(selected_formulas) >= k:
                    break
                remaining = k - len(selected_formulas)
            else:
                remaining = sum(
                    max(0, validated_depth_targets[d] - per_depth_counts[d])
                    for d in validated_depth_targets
                )
                if remaining <= 0:
                    break

            attempts += 1
            sample_batch_size = 51200
            sampled = kernel.sample_dataset_formulas_kernel(
                k=sample_batch_size,
                p_leaf_range=p_leaf_range,
                max_depth=max_depth,
                force_tree=False,
            )

            for phi in sampled:
                if validated_depth_targets is None:
                    if len(selected_formulas) >= k:
                        break
                else:
                    outstanding = any(
                        per_depth_counts[d] < validated_depth_targets[d]
                        for d in validated_depth_targets
                    )
                    if not outstanding:
                        break

                phi_depth = phi.depth()
                if min_depth is not None and phi_depth < min_depth:
                    rejected_depth += 1
                    continue

                if validated_depth_targets is not None:
                    if phi_depth not in validated_depth_targets:
                        rejected_depth += 1
                        continue
                    if per_depth_counts[phi_depth] >= validated_depth_targets[phi_depth]:
                        rejected_filled_depth += 1
                        continue

                phi_str = str(phi)
                if phi_str in exclude:
                    rejected_excluded += 1
                    continue

                if dedupe and phi_str in selected_strs:
                    rejected_duplicate += 1
                    continue

                if validated_depth_targets is None:
                    selected_formulas.append(phi)
                else:
                    per_depth_selected[phi_depth].append(phi)
                    per_depth_counts[phi_depth] += 1

                if dedupe:
                    selected_strs.add(phi_str)

        if validated_depth_targets is not None:
            for depth in sorted(validated_depth_targets.keys()):
                selected_formulas.extend(per_depth_selected[depth])

        if len(selected_formulas) < k:
            missing_by_depth: dict[int, int] | None = None
            if validated_depth_targets is not None:
                missing_by_depth = {
                    depth: max(0, validated_depth_targets[depth] - per_depth_counts[depth])
                    for depth in sorted(validated_depth_targets.keys())
                }
            raise ValueError(
                "Could not sample enough validation formulas after applying exclusions. "
                f"Requested={k}, sampled={len(selected_formulas)}, attempts={attempts}, "
                f"rejected_excluded={rejected_excluded}, rejected_depth={rejected_depth}, "
                f"rejected_duplicate={rejected_duplicate}, "
                f"rejected_filled_depth={rejected_filled_depth}, "
                f"missing_by_depth={missing_by_depth}."
            )

        dataset = LTLDataset(
            store_formula_str=store_formula_str,
            store_satisfaction=store_satisfaction,
            satisfaction_batch_size=satisfaction_batch_size,
            satisfaction_time_index=satisfaction_time_index,
        )
        dataset.construct_dataset_from_list(selected_formulas, kernel)
        dataset.metadata.update({
            "source": "kernel_excluding",
            "requested_k": k,
            "sampled_k": len(dataset),
            "p_leaf_range": p_leaf_range,
            "max_depth": max_depth,
            "min_depth": min_depth,
            "depth_targets": validated_depth_targets,
            "sampled_per_depth": per_depth_counts if validated_depth_targets is not None else None,
            "dedupe": dedupe,
            "exclude_count": len(exclude),
            "max_sampling_attempts": max_sampling_attempts,
            "sampling_attempts_used": attempts,
            "rejected_excluded": rejected_excluded,
            "rejected_depth": rejected_depth,
            "rejected_duplicate": rejected_duplicate,
            "rejected_filled_depth": rejected_filled_depth,
            "kernel_T": kernel.T,
            "kernel_AP": kernel.AP,
            "kernel_seed": kernel.seed,
        })
        return dataset



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


    @staticmethod
    def construct_finetuning_mutation_dataset(
        kernel: LTLKernel,
        stage_train_dirs: list[str],
        sample_count: int = 20000,
        equivalent_mutations_per_formula: int = 2,
        near_miss_mutations_per_formula: int = 1,
        exclude_formula_strs: set[str] | None = None,
        store_formula_str: bool = True,
        store_satisfaction: bool = True,
        satisfaction_batch_size: int = 10240,
        satisfaction_time_index: int = 0,
        seed: int | None = None,
    ) -> LTLDataset:
        """
        Build a fine-tuning dataset from cumulative curriculum stages.

        Sampling strategy:
        - Each path in `stage_train_dirs` is expected to be a cumulative stage dataset
          (stage i contains stage i-1 + newly sampled formulas).
        - The method derives per-stage "new" slices by length differences and samples
          approximately equally from each stage's new slice, matching the requested
          base-2/logarithmic balancing across doubling stage sizes.

        Mutation strategy per sampled base formula:
        - apply `equivalent_mutations_per_formula` random semantic-equivalent rewrites,
          each chosen uniformly among rewrites available for the current formula;
        - then create `near_miss_mutations_per_formula` variants by adding one random
          negation at a random AST location.
        """
        if len(stage_train_dirs) == 0:
            raise ValueError("stage_train_dirs must contain at least one cumulative stage directory")
        if sample_count <= 0:
            raise ValueError("sample_count must be > 0")
        if equivalent_mutations_per_formula < 0:
            raise ValueError("equivalent_mutations_per_formula must be >= 0")
        if near_miss_mutations_per_formula < 0:
            raise ValueError("near_miss_mutations_per_formula must be >= 0")

        exclude = set(exclude_formula_strs) if exclude_formula_strs is not None else set()
        print('Loaded `exclude` set.')
        stage_datasets = [LTLDataset.load(path, load_satisfactions=False) for path in stage_train_dirs]
        print('Loaded stage datasets.')

        stage_new_pools: list[list[Formula]] = []
        stage_new_sizes: list[int] = []
        prev_len = 0
        for stage_idx, dataset in enumerate(stage_datasets):
            curr_len = len(dataset)
            if curr_len < prev_len:
                raise ValueError(
                    f"Stage dataset at index {stage_idx} is smaller than previous stage "
                    f"({curr_len} < {prev_len}). Expected cumulative stage datasets."
                )
            stage_new = [
                phi for phi in dataset.formulas[prev_len:curr_len]
                if str(phi) not in exclude
            ]
            stage_new_pools.append(stage_new)
            stage_new_sizes.append(len(stage_new))
            prev_len = curr_len

        total_available = sum(stage_new_sizes)
        if total_available < sample_count:
            raise ValueError(
                f"Unable to sample {sample_count} base formulas after exclusions; "
                f"only {total_available} are available."
            )

        n_stages = len(stage_new_pools)
        base_quota = sample_count // n_stages
        quotas = [base_quota for _ in range(n_stages)]
        for i in range(sample_count % n_stages):
            quotas[i] += 1

        sampled_per_stage = [min(quotas[i], stage_new_sizes[i]) for i in range(n_stages)]
        remaining = sample_count - sum(sampled_per_stage)
        capacities = [stage_new_sizes[i] - sampled_per_stage[i] for i in range(n_stages)]
        while remaining > 0 and sum(capacities) > 0:
            for i in range(n_stages):
                if remaining == 0:
                    break
                if capacities[i] <= 0:
                    continue
                sampled_per_stage[i] += 1
                capacities[i] -= 1
                remaining -= 1

        if remaining > 0:
            raise ValueError(
                f"Unable to sample {sample_count} formulas without replacement from stage pools; "
                f"maximum available is {sum(stage_new_sizes)}"
            )
        print(f'Produced stage new pools. Their sizes are {stage_new_sizes} and number of samples that will end up in the final dataset are {sampled_per_stage}')

        seed_value = int(seed) if seed is not None else int(getattr(kernel, "seed", 0) or 0)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed_value)

        stage_perms: list[list[int]] = [
            torch.randperm(len(pool), generator=rng).tolist()
            for pool in stage_new_pools
        ]
        stage_drawn_counts = [0 for _ in stage_new_pools]
        print('Produced stage perms.')

        def _draw_from_stage(stage_idx: int, n_draw: int) -> list[Formula]:
            if n_draw <= 0:
                return []
            perm = stage_perms[stage_idx]
            ptr = stage_drawn_counts[stage_idx]
            available = len(perm) - ptr
            take = min(n_draw, available)
            if take <= 0:
                return []

            picked_indices = perm[ptr : ptr + take]
            stage_drawn_counts[stage_idx] += take
            pool = stage_new_pools[stage_idx]
            return [pool[i] for i in picked_indices]

        def _sample_without_replacement(candidates: list[Formula], n_take: int) -> list[Formula]:
            if n_take <= 0 or len(candidates) == 0:
                return []
            if n_take >= len(candidates):
                return list(candidates)
            ids = torch.randperm(len(candidates), generator=rng)[:n_take].tolist()
            return [candidates[i] for i in ids]

        def _sample_excluding_with_oversampling(
            candidates: list[Formula],
            n_take: int,
            oversample_factor: int = 4,
        ) -> list[Formula]:
            """
            Sample candidates without replacement, then filter excluded formulas.
            This avoids calling str(phi) for every candidate when candidate pools are large.
            """
            if n_take <= 0 or len(candidates) == 0:
                return []
            n_draw = min(len(candidates), max(n_take, n_take * oversample_factor))
            proposed = _sample_without_replacement(candidates, n_draw)
            selected: list[Formula] = []
            for phi in proposed:
                if str(phi) in exclude:
                    continue
                selected.append(phi)
                if len(selected) >= n_take:
                    break
            return selected

        sampled_base_formulas: list[Formula] = []
        for stage_idx, n_pick in enumerate(sampled_per_stage):
            sampled_base_formulas.extend(_draw_from_stage(stage_idx, n_pick))
            print(f'Finished sampling formulas for stage{stage_idx+1}.')

        total_sampled = len(sampled_base_formulas)
        if total_sampled != sample_count:
            raise ValueError(
                f"Unexpected sampled base count {total_sampled}; expected exactly {sample_count}."
            )

        target_equivalent = sample_count * equivalent_mutations_per_formula
        target_near_miss = sample_count * near_miss_mutations_per_formula

        mutated_formulas: list[Formula] = []
        equivalent_formulas: list[Formula] = []
        near_miss_formulas: list[Formula] = []
        equivalent_success = 0
        near_miss_success = 0
        base_formulas_processed = 0

        def _process_base_formula(base_formula: Formula) -> None:
            nonlocal equivalent_success, near_miss_success

            remaining_eq = target_equivalent - len(equivalent_formulas)
            if equivalent_mutations_per_formula > 0 and remaining_eq > 0:
                eq_budget = min(equivalent_mutations_per_formula, remaining_eq)
                eq_candidates = list_semantically_equivalent_transformations(base_formula)
                eq_selected = _sample_excluding_with_oversampling(eq_candidates, eq_budget)
                equivalent_formulas.extend(eq_selected)
                mutated_formulas.extend(eq_selected) 
                equivalent_success += len(eq_selected)

            remaining_nm = target_near_miss - len(near_miss_formulas)
            if near_miss_mutations_per_formula > 0 and remaining_nm > 0:
                nm_budget = min(near_miss_mutations_per_formula, remaining_nm)
                nm_candidates = list_negation_insertions(base_formula)
                nm_selected = _sample_excluding_with_oversampling(nm_candidates, nm_budget)
                near_miss_formulas.extend(nm_selected)
                mutated_formulas.extend(nm_selected)
                near_miss_success += len(nm_selected)

        for base_formula in sampled_base_formulas:
            _process_base_formula(base_formula)
            base_formulas_processed += 1
            if base_formulas_processed >= 20000:
                print(f'The number of processed formulas is: {base_formulas_processed}')
            if len(equivalent_formulas) >= target_equivalent and len(near_miss_formulas) >= target_near_miss:
                print(f'Breaking out of the initial base_formula processing loop with n_equiv={len(equivalent_formulas)} and n_near_miss={len(near_miss_formulas)}')
                break

        stage_cursor = 0
        while len(equivalent_formulas) < target_equivalent or len(near_miss_formulas) < target_near_miss:
            print(f'Sampling extra formulas since {len(equivalent_formulas)}<{target_equivalent} OR {len(near_miss_formulas)}<{target_near_miss}.')
            picked_extra: Formula | None = None
            for _ in range(n_stages):
                stage_idx = stage_cursor
                stage_cursor = (stage_cursor + 1) % n_stages
                drawn = _draw_from_stage(stage_idx, 1)
                if drawn:
                    picked_extra = drawn[0]
                    break

            if picked_extra is None:
                break

            _process_base_formula(picked_extra)
            base_formulas_processed += 1
            print(f'The number of processed formulas is: {base_formulas_processed}')

        missing_eq = target_equivalent - len(equivalent_formulas)
        missing_nm = target_near_miss - len(near_miss_formulas)
        if missing_eq > 0 or missing_nm > 0:
            raise ValueError(
                "Could not satisfy requested mutation counts after applying exclusions. "
                f"Missing equivalent={missing_eq}, near_miss={missing_nm}. "
                "Consider relaxing exclusions, increasing stage pools, or reducing mutation counts."
            )

        dataset = LTLDataset(
            store_formula_str=store_formula_str,
            store_satisfaction=store_satisfaction,
            satisfaction_batch_size=satisfaction_batch_size,
            satisfaction_time_index=satisfaction_time_index,
        )
        dataset._reset_storage()

        print('started evaluating')
        embedding_cache: dict[str, torch.Tensor] = {}
        satisfaction_cache: dict[str, torch.Tensor] = {}
        for i, phi in enumerate(mutated_formulas):
            phi_str = str(phi)
            if phi_str not in embedding_cache:
                phi_sats = kernel._evaluate_formula_on_traces(
                    formula=phi,
                    batch_size=satisfaction_batch_size,
                    time_index=satisfaction_time_index,
                )
                embedding_cache[phi_str] = kernel.compute_embedding_from_satisfaction(phi_sats, move_to_cpu=True)
                if store_satisfaction:
                    satisfaction_cache[phi_str] = phi_sats.clone().to("cpu")

            emb = embedding_cache[phi_str]
            sats_to_store = satisfaction_cache.get(phi_str) if store_satisfaction else None
            dataset._append_entry(phi, emb, sats_to_store)
            if (i+1) % 1000 == 1:
                print(f'Number of formulas evaluated={i+1}')

        dataset.metadata.update({
            "source": "finetune_mutation",
            "stage_train_dirs": stage_train_dirs,
            "sample_count_requested": sample_count,
            "sample_count_actual": total_sampled,
            "sampled_per_stage_initial": sampled_per_stage,
            "sampled_per_stage_total": stage_drawn_counts,
            "base_formulas_processed": base_formulas_processed,
            "extra_base_samples_drawn": max(0, base_formulas_processed - sample_count),
            "equivalent_mutations_per_formula": equivalent_mutations_per_formula,
            "near_miss_mutations_per_formula": near_miss_mutations_per_formula,
            "equivalent_target": target_equivalent,
            "near_miss_target": target_near_miss,
            "equivalent_mutations_generated": equivalent_success,
            "near_miss_mutations_generated": near_miss_success,
            "mutated_count": len(mutated_formulas),
            "excluded_formula_count": len(exclude),
            "kernel_T": kernel.T,
            "kernel_AP": kernel.AP,
            "kernel_seed": kernel.seed,
            "seed": seed_value,
        })

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

        prev_len = 0
        if prev_dirpath is not None:
            prev_metadata_path = os.path.join(prev_dirpath, "metadata.json")
            if os.path.exists(prev_metadata_path):
                with open(prev_metadata_path, "r", encoding="utf-8") as fp:
                    prev_metadata = json.load(fp)
                prev_len = prev_metadata.get("size", 0)
                if prev_len == 0:
                    raise ArithmeticError(f"The metadata at {prev_metadata_path} reports a size of 0. Please inspect the datasets manually.")
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

        part_path = os.path.join(dirpath, f"satisfactions.part{rank}.pt")
        num_new_in_chunk = len(formulas_chunk)
        
        if num_new_in_chunk > 0:
            # Pre-allocate chunk tensor on GPU
            num_traces = kernel.traces.size(0)
            sats_tensor = torch.empty((num_new_in_chunk, num_traces), dtype=torch.bool, device="cuda")
            
            for i, phi in enumerate(formulas_chunk):
                phi_sats = kernel._evaluate_formula_on_traces(
                    formula=phi,
                    batch_size=batch_size,
                    time_index=time_index,
                )
                # Copy directly into the pre-allocated buffer
                sats_tensor[i] = phi_sats.to(dtype=torch.bool)
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
            num_traces = kernel.traces.size(0)
            # Pre-allocate full result tensor on CPU to avoid multiple large copies in RAM
            sats_tensor = torch.empty((total, num_traces), dtype=torch.bool, device="cpu")

            # 1. Copy previous satisfactions if they exist
            if prev_len > 0:
                prev_sats_path = os.path.join(prev_dirpath, "satisfactions.pt")
                if os.path.exists(prev_sats_path):
                    print(f"Loading previous satisfactions (mmap) from {prev_sats_path}...")
                    # mmap=True allows loading without immediate allocation of full RAM
                    prev_data = torch.load(prev_sats_path, map_location="cpu", mmap=True)
                    sats_tensor[:prev_len] = prev_data
                    del prev_data
                else:
                    raise FileNotFoundError(f"Previous satisfactions.pt not found in {prev_dirpath}")

            # 2. Load and copy new chunks sequentially
            current_idx = prev_len
            print(f"Aggregating {world_size} satisfaction parts sequentially...")
            for r in range(world_size):
                part_path = os.path.join(dirpath, f"satisfactions.part{r}.pt")
                while not os.path.exists(part_path):
                    time.sleep(1)
                
                # Load chunk, copy to pre-allocated slice, then free memory
                part_data = torch.load(part_path, map_location="cpu", mmap=True)
                num_in_part = part_data.size(0)
                if num_in_part > 0:
                    sats_tensor[current_idx : current_idx + num_in_part] = part_data
                    current_idx += num_in_part
                
                del part_data
                os.remove(part_path)

            # 3. Save final consolidated tensor
            torch.save(sats_tensor, satisfactions_path)
            del sats_tensor

            metadata["store_satisfaction"] = True
            metadata["has_satisfactions"] = True
            metadata["satisfaction_batch_size"] = batch_size
            metadata["satisfaction_time_index"] = time_index

            with open(metadata_path, "w", encoding="utf-8") as fp:
                json.dump(metadata, fp, indent=2)


    
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
    def load(
        cls,
        dirpath: str,
        load_satisfactions: bool = True,
        satisfactions_mmap: bool = False,
    ) -> "LTLDataset":
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

        if load_satisfactions and metadata.get("has_satisfactions") and os.path.exists(satisfactions_path):
            load_kwargs = {"map_location": "cpu"}
            if satisfactions_mmap:
                load_kwargs["mmap"] = True
            dataset.satisfactions = torch.load(satisfactions_path, **load_kwargs)
        else:
            dataset.satisfactions = None

        dataset.metadata = metadata.get("extra_metadata", {})
        return dataset