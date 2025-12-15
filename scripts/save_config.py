import os
from tokenizer_pretrained_class import LTLTokenizer
from config_class import LTLConfig

def main():
    # Hyperparameters
    AP      = 5
    m       = 1024
        
    # Create output directory
    output_dir = "artifacts/config"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer (adjust n_ap based on your needs)
    tokenizer = LTLTokenizer(n_ap=AP)

    
    print("Finished train and eval dataset construction.")

    # Create model configuration and model
    config = LTLConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=m,  # must match kernel's anchor set size (m)
        n_head=16,        # must divide kernel.m (!!!)
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    config.save_pretrained(output_dir)
    print(f"Config saved to {output_dir}/config.json")
    
if __name__ == "__main__":
    main()