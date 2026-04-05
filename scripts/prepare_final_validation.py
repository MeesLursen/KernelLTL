"""Generate a master validation set that is disjoint from all curriculum stages."""

import argparse
import os
import torch
from dataset_class import LTLDataset
from kernel_class import LTLKernel
from formula_class import Formula
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Create a stratified master validation set.")
    parser.add_argument("--kernel-dir", required=True, help="Path to kernel")
    parser.add_argument("--latest-train-dir", required=True, help="Path to the latest cumulative train dataset")
    parser.add_argument("--latest-eval-dir", required=True, help="Path to the latest cumulative eval dataset")
    parser.add_argument("--output-dir", required=True, help="Where to save the validation set")
    parser.add_argument("--k-per-depth", type=int, default=500, help="Number of formulas to sample for each depth level")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum depth to sample up to")
    parser.add_argument("--p-leaf-range", nargs=2, type=float, default=[0.1, 0.5], help="P_leaf range for sampling")
    parser.add_argument("--batch-size", type=int, default=10240, help="Batch size for satisfaction calculation")
    return parser.parse_args()

def main():
    args = parse_args()
    kernel = LTLKernel.load(args.kernel_dir)
    
    print("Loading seen formulas from latest stages...")
    train_ds = LTLDataset.load(args.latest_train_dir)
    eval_ds = LTLDataset.load(args.latest_eval_dir)
    
    seen_formulas = set()
    for f in train_ds.formulas:
        seen_formulas.add(str(f))
    for f in eval_ds.formulas:
        seen_formulas.add(str(f))
    
    print(f"Total seen formulas to exclude: {len(seen_formulas)}")

    val_dataset = LTLDataset(
        store_formula_str=True,
        store_satisfaction=True,
        satisfaction_batch_size=args.batch_size
    )
    val_dataset._reset_storage()

    # Stratified sampling
    for depth in range(1, args.max_depth + 1):
        print(f"Sampling for depth {depth}...")
        found_at_depth = 0
        attempts = 0
        max_attempts = args.k_per_depth * 50 # Avoid infinite loops if space is exhausted
        
        while found_at_depth < args.k_per_depth and attempts < max_attempts:
            attempts += 1
            # Sample a batch of formulas at this depth
            # Using force_tree=False to allow variety, but max_depth=depth
            samples = kernel.sample_dataset_formulas_kernel(
                k=max(100, args.k_per_depth), 
                p_leaf_range=tuple(args.p_leaf_range), 
                max_depth=depth, 
                force_tree=False
            )
            
            for phi in samples:
                if found_at_depth >= args.k_per_depth:
                    break
                
                # Strict depth check + exclusion check
                if phi.depth() == depth and str(phi) not in seen_formulas:
                    phi_str = str(phi)
                    seen_formulas.add(phi_str) # Don't duplicate within validation set
                    
                    # Compute satisfaction and embedding
                    phi_sats = kernel._evaluate_formula_on_traces(
                        formula=phi,
                        batch_size=args.batch_size,
                        time_index=0
                    )
                    emb = kernel.compute_embedding_from_satisfaction(phi_sats, move_to_cpu=True)
                    
                    val_dataset._append_entry(phi, emb, phi_sats.to('cpu'))
                    found_at_depth += 1
        
        print(f"  Finished depth {depth}: Found {found_at_depth} new formulas.")

    val_dataset.metadata.update({
        "source": "master_validation_generator",
        "max_depth": args.max_depth,
        "k_per_depth": args.k_per_depth,
        "excluded_base_count": len(seen_formulas) - len(val_dataset)
    })
    
    val_dataset.save(args.output_dir)
    print(f"Master validation set saved to {args.output_dir} with {len(val_dataset)} formulas.")

if __name__ == "__main__":
    main()