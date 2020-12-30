#! /usr/bin/env python3
__AUTHORS__ = [
  ("CS17BTECH11044", "YASH KHASBAGE"),
  ("CS17BTECH11029", "PUNEET MANGLA")
]

import json
import pandas as pd
from tokenizers import BertWordPieceTokenizer

# Initialize an empty BERT tokenizer
tokenizer = BertWordPieceTokenizer(
  clean_text=True,
  handle_chinese_chars=False,
  strip_accents=False,
  lowercase=True,
)

# prepare text files to train vocab on them
files = ['input.txt']

# train BERT tokenizer
tokenizer.train(
  files,
  vocab_size=30000,
  min_frequency=10,
  show_progress=True,
  special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
  limit_alphabet=1000,
  wordpieces_prefix="##"
)

# save vocabulary
tokenizer.save('./udc_vocab.txt')

# print vocabulary
f = open('./udc_vocab.txt')
jf = json.load(f)
vocab = list(jf['model']['vocab'].keys())
vocab = '\n'.join(vocab)
print(vocab)