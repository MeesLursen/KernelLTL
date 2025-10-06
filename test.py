import torch
from tokenizer_class import LTLTokenizer
from formula_class import Formula, sample_formulas

AP = 5

device = 'cpu'
rng = torch.Generator(device=device).manual_seed(548)
tokenizer = LTLTokenizer(AP)

formula = sample_formulas(1,0.45,100000,AP,True,rng,device)[0]
print(formula)

encoded_formula = tokenizer.encode(formula.__str__(), 35)
print(encoded_formula)

decoded_formula = tokenizer.decode(encoded_formula)
print(decoded_formula)

formula