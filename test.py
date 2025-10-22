from tokenizer_pretrained_class import LTLTokenizer

tokenizer = LTLTokenizer(n_ap = 5)


print(tokenizer._token_to_id('eos'))