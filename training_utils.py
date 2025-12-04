import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from formula_utils import str_to_formula
from kernel_class import LTLKernel
from tokenizer_pretrained_class import LTLTokenizer
from model_class import LTLModel


class SemanticEvaluationCallback(TrainerCallback):
    """
    Custom callback for evaluating semantic similarity between generated and target formulas.
    Computes kernel embeddings of generated formulas and compares with target embeddings.
    """
    def __init__(self, 
                 kernel: LTLKernel,
                 tokenizer: LTLTokenizer,
                 eval_dataset,
                 kernel_eval_batch_size: int = 512,
                 kernel_time_index: int = 0):
        """
        Args:
            kernel: LTLKernel instance for computing semantic embeddings
            tokenizer: LTLTokenizer for decoding generated sequences
        """
        self.kernel = kernel
        self.tokenizer: LTLTokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.kernel_eval_batch_size = kernel_eval_batch_size
        self.kernel_time_index = kernel_time_index
        
        # metrics history
        self.epochs: list[int] = []
        self.semantic_distances: list[float] = []
        self.exact_matches: list[float] = []

    def on_epoch_end(self,
                     args: TrainingArguments,
                     state: TrainerState,
                     control: TrainerControl,
                     model: LTLModel,
                     **kwargs):
        
        # Only execute on global rank 0 to avoid duplicate expensive evaluation under DDP.
        local_rank = getattr(args, "local_rank", -1)
        if local_rank not in (-1, 0):
            return
        
        eval_dataset = self.eval_dataset
        if eval_dataset is None:
            return

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.dataloader_num_workers,
            collate_fn=lambda batch : self.tokenizer.collate_batch(batch, model.config.n_positions, include_metadata=True),
            pin_memory=args.dataloader_pin_memory,
            shuffle=False
        )

        total_distance = 0.0
        exact_matches = 0
        total_samples = 0
        exact_matches_strs = []
        mismatch_strs = []
        invalid_syntax_strs = []

        model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'] 
                target_embeddings = batch['semantic_embeddings'].to(model.device, non_blocking=True)
                attention_mask = batch['attention_mask']
                target_formulas = batch.get('target_formulas')
                target_formula_strs = batch.get('target_formula_strs')
                target_satisfaction = batch.get('target_satisfaction')
                if target_satisfaction is not None:
                    target_satisfaction = target_satisfaction.to(self.kernel.device)
                
                if target_formula_strs is not None:
                    target_strs = target_formula_strs
                else:
                    target_strs = []
                    for ids, mask in zip(input_ids, attention_mask):
                        valid_ids = ids[mask.bool()].tolist()
                        target_strs.append(self.tokenizer.decode(valid_ids, skip_special_tokens=True))

                if target_formulas is None:
                    target_formulas = [str_to_formula(s) for s in target_strs]

                batch_size = target_embeddings.size(0)
                total_samples += batch_size

                generated_ids = model.generate(
                    semantic_embeddings=target_embeddings,
                    max_length=model.config.n_positions,
                    num_beams=1,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                generated_strs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for i in range(batch_size):
                    generated_str = generated_strs[i]
                    target_str = target_strs[i]
                    target_formula = target_formulas[i]
                    if target_satisfaction is not None:
                        target_sats = target_satisfaction[i]
                    else:
                        target_sats = self.kernel._evaluate_formula_on_traces(
                            formula=target_formula,
                            batch_size=self.kernel_eval_batch_size,
                            time_index=self.kernel_time_index
                        )

                    try:
                        generated_formula = str_to_formula(generated_str)
                        generated_sats = self.kernel._evaluate_formula_on_traces(
                            formula=generated_formula,
                            batch_size=self.kernel_eval_batch_size,
                            time_index=self.kernel_time_index
                        )

                        xor = torch.logical_xor(target_sats, generated_sats)
                        distance = xor.to(dtype=torch.float32).mean().item()
                        total_distance += distance
                        
                        if str(generated_formula) == str(target_formula):
                            exact_matches += 1
                            exact_matches_strs.append(generated_str)
                        else:
                            mismatch_strs.append((target_str,generated_str))

                    except Exception:
                        # Penalize for invalid formula by adding max distance
                        total_distance += 1.0
                        invalid_syntax_strs.append(generated_str)

        avg_distance = total_distance / total_samples if total_samples > 0 else 0
        exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0
        
        self.epochs.append(state.epoch)
        self.semantic_distances.append(avg_distance)
        self.exact_matches.append(exact_match_rate)
        
        print(f"\n  Epoch {state.epoch}:")
        print(f"  Semantic Distance: {avg_distance:.4f}")
        print(f"  Exact Match Rate: {exact_match_rate:.4f}")
        print(f"  Exact Match Strings: {exact_matches_strs}")
        print(f"  Mismatch Strings: {mismatch_strs}")
        print(f"  Invalid Syntax Strs: {invalid_syntax_strs}")


