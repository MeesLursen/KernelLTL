import math
import os
import torch

from transformers import TrainingArguments, Trainer
from tokenizer_pretrained_class import LTLTokenizer
from kernel_class import LTLKernel
from dataset_class import LTLDataset
from model_class import LTLModel
from config_class import LTLConfig
from training_utils import SemanticEvaluationCallback

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Hyperparameters
    num_epochs = 10

    learning_rate = 5e-5

    T       = 20
    AP      = 5
    seed    = 1

    eps     = 0.01
    delta   = 1 - 0.99
    N       = math.ceil((2 / eps**2) * math.log(2 / delta))

    m       = 1024
        
    # Create output directory
    output_dir = "ltl_model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer (adjust n_ap based on your needs)
    tokenizer = LTLTokenizer(n_ap=AP)
    
    # Initialize kernel for semantic embeddings
    kernel = LTLKernel(T, AP, seed)  # adjust T and AP as needed
    kernel.sample_traces_kernel(N)  # adjust N based on your needs
    kernel.sample_anchor_formulas_kernel(m)  # m should match model's n_embd
    kernel.build_F()
    
    # Create datasets
    train_dataset = LTLDataset()
    train_dataset.construct_dataset_from_kernel(
        kernel=kernel,
        k=78000,  # adjust dataset size as needed
        p_leaf=0.45,
        max_depth=1000,
        batch_size=10240
    )
    
    eval_dataset = LTLDataset()
    eval_dataset.construct_dataset_from_kernel(
        kernel=kernel,
        k=1000,  # smaller validation set
        p_leaf=0.45,
        max_depth=1000,
        batch_size=10240
    )
    
    # Create model configuration and model
    config = LTLConfig(
        tokenizer=tokenizer,
        n_embd=m  # must match kernel's anchor set size (m)
    )
    
    model = LTLModel(config, semantic_emb_dim=m)  # semantic_emb_dim must match kernel's anchor set size
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_safetensors=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        ddp_find_unused_parameters=False
    )
    
    # Initialize callback
    semantic_callback = SemanticEvaluationCallback(
        kernel=kernel,
        tokenizer=tokenizer
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=lambda batch : tokenizer.collate_batch(batch, model.config.n_positions),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[semantic_callback]
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    
if __name__ == "__main__":
    main()