import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from formula_class import Formula
from formula_utils import str_to_formula
from kernel_class import LTLKernel
from tokenizer_class import LTLTokenizer
from model_class import LTLModel


class SemanticEvaluationCallback(TrainerCallback):
    """
    Custom callback for evaluating semantic similarity between generated and target formulas.
    Computes kernel embeddings of generated formulas and compares with target embeddings.
    """
    def __init__(self, 
                 kernel: LTLKernel,
                 tokenizer: LTLTokenizer,
                 num_samples: int = 16):
        """
        Args:
            kernel: LTLKernel instance for computing semantic embeddings
            tokenizer: LTLTokenizer for decoding generated sequences
            num_samples: Number of validation samples to evaluate
        """
        self.kernel = kernel
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        
        # metrics history
        self.epochs: list[int] = []
        self.semantic_distances: list[float] = []
        self.exact_matches: list[float] = []

    def on_epoch_end(self,
                    args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl,
                    model: LTLModel | None = None,
                    **kwargs):
        """Run semantic evaluation at the end of each epoch."""
        if not model or not hasattr(state, "eval_dataloader"):
            return
            
        model.eval()
        with torch.no_grad():
            for batch in state.eval_dataloader:
                batch_size = len(batch["encoder_embeddings"])
                
                # extract inputs 
                encoder_embeddings = batch["encoder_embeddings"][:batch_size].to(model.device)
                input_ids = batch["input_ids"][:batch_size].to(model.device)
                attention_mask = batch["attention_mask"][:batch_size].to(model.device)
                
                # generate formula sequences
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    semantic_embeddings=encoder_embeddings,
                    max_length = model.config.n_positions,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_id,
                    eos_token_id=self.tokenizer.eos_id,
                )
                
                # decode generated sequences
                generated_formulas = []
                for ids in generated_ids:
                    formula_str = self.tokenizer.decode(ids, skip_special_tokens=True)
                    try:
                        # you need to implement parse_formula to convert string to Formula object
                        formula = str_to_formula(formula_str)
                        generated_formulas.append(formula)
                    except:
                        # if parsing fails, skip this sample in evaluation
                        continue
                
                if not generated_formulas:
                    return
                    
                # compute embeddings for generated formulas
                generated_embeddings = []
                for formula in generated_formulas:
                    emb = self.kernel.compute_formula_embedding_normalized(
                        formula, 
                        device=model.device
                    )
                    generated_embeddings.append(emb)
                
                generated_embeddings = torch.stack(generated_embeddings)
                target_embeddings = encoder_embeddings[:len(generated_embeddings)]
                
                # compute cosine similarity between embeddings
                similarities = torch.nn.functional.cosine_similarity(
                    generated_embeddings,
                    target_embeddings,
                    dim=1
                )
                semantic_distance = 1 - similarities.mean().item()
                
                # compute exact matches
                exact_match = (similarities > 0.999).float().mean().item()
                
                # log metrics
                self.epochs.append(state.epoch)
                self.semantic_distances.append(semantic_distance) 
                self.exact_matches.append(exact_match)
                
                print(f"\nEpoch {state.epoch}:")
                print(f"Average semantic distance: {semantic_distance:.4f}")
                print(f"Exact match ratio: {exact_match:.4f}")
        
        model.train()
