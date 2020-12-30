#!/usr/bin/env python3
__AUTHORS__ = [
    ('PUNEET MANGLA', 'CS17BTECH11029'),
    ('YASH KHASBAGE', 'CS17BTECH11044')
]

import os
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
import os.path as osp

SAVE_DIR = '/DATA1/puneet/IR'

# parser
def parse_args():
    parser = argparse.ArgumentParser(
        description= 'IR Based Chatbot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # dataset and architecture
    parser.add_argument("--dataset", default="udc", type=str,
                        help="dataset")
    parser.add_argument("--exp_name", default=None, type=str,
                        help="dataset")
    parser.add_argument("--config_name", default='bert-base-uncased', type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--split", default="train", type=str, choices=['train', 'test', 'valid'],
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default='bert-base-uncased', type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str)
    parser.add_argument("--dataset_root", type=str, default='./../', 
                        help='dataset root, should contain train.csv, test.csv, valid.csv inside it.')


    parser.add_argument("--resume", action='store_true',
                        help="Train using mixup")

    # hyperparameters
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_freq', type=int, default=10,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--iter', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--cuda', type=int, help='gpu number')
    
    return parser.parse_args()

# automated resume
def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [x for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

# resume given file details
def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file
