#!/usr/bin/env python3

__AUTHORS__ = [
    ('CS17BTECH11044', 'YASH KHASBAGE'),
    ('CS17BTECH11029', 'PUNEET MANGLA')
]

import os
import re
import json
import torch
import pandas as pd 
import os.path as osp
import torch.utils.data as data
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

# get stop words
stop_words = set(stopwords.words('english')) 

# dataset
class UDC(data.Dataset):
    def __init__(self, root, tokenizer, split='train', max_length=(86, 17)):
        
        # maxlen (86,17)
        self.root = os.path.expanduser(root)
        self.split = split
        self.split_json = os.path.join(root, self.split + '.csv')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        print('Data file : ',self.split_json)

        # read dataset csv
        assert osp.exists(self.split_json), self.split_json + " not found"
        self.dataset = pd.read_csv(self.split_json)
        
        # use subset of dataset due to computational constraints
        if split == 'train':
            self.dataset = self.dataset.iloc[:50000, :]
        else:
            self.dataset = self.dataset.iloc[:5000, :]
        print('dataset size', self.dataset.shape)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.split == 'train':
            # sample from dataset
            context, response, label = self.dataset.iloc[index]
            # tokenize using nltk tokenizer
            context, response  = word_tokenize(context), word_tokenize(response)

            # encode using bert tokenizer
            context = self.tokenizer.encode_plus(
                            context, 
                            add_special_tokens=True,          # Sentence to encode.
                            max_length = self.max_length[0],  # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,     # Construct attn. masks.
                            truncation=True,
                            return_tensors = 'pt',            # Return pytorch tensors.
                    )
            response = self.tokenizer.encode_plus(
                            response,  
                            add_special_tokens=True,          # Sentence to encode.
                            max_length = self.max_length[1],  # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,     # Construct attn. masks.
                            truncation=True,
                            return_tensors = 'pt',            # Return pytorch tensors.
                    )
    
            label = int(label)
            return [
                context['input_ids'].squeeze(0), 
                context['attention_mask'].squeeze(0), 
                response['input_ids'].squeeze(0), 
                response['attention_mask'].squeeze(0), 
                label
            ]
        else:

            # similar to train set
            row = self.dataset.iloc[index].tolist()
            context = row[0]
            responses = row[1:]
            assert len(responses) == 10, 'number of responses: {}'.format(len(responses))
            
            # tokenize with nltk toeknizer
            context = word_tokenize(context)
            responses = [word_tokenize(response) for response in responses]

            # use bert tokenizer for encoding
            context = self.tokenizer.encode_plus(
                            context,                    
                            add_special_tokens = True, 
                            max_length = self.max_length[0],     
                            pad_to_max_length = True,
                            return_attention_mask = True,  
                            truncation=True,
                            return_tensors = 'pt',
                    )            

            responses = [
                self.tokenizer.encode_plus(
                    r,                   
                    add_special_tokens = True, 
                    max_length = self.max_length[1],      
                    pad_to_max_length = True,
                    return_attention_mask = True,
                    truncation=True,
                    return_tensors = 'pt',   
                ) for r in responses
            ]

            response_inputs_ids = torch.cat([r['input_ids'] for r in responses])
            response_attention_mask = torch.cat([r['attention_mask'] for r in responses])
            
            return [
                context['input_ids'].squeeze(0),
                context['attention_mask'].squeeze(0),
                response_inputs_ids,
                response_attention_mask,
                row
            ]
