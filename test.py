import torch.nn as nn
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
print(hasattr(TrainerCallback, 'eval_dataloader'))  # Should print: True