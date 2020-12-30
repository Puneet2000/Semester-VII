#!/usr/bin/env python3
__AUTHORS__ = [
    ('PUNEET MANGLA', 'CS17BTECH11029'),
    ('YASH KHASBAGE', 'CS17BTECH11044')
]

import os
import pdb
import torch
import numpy as np
import pickle as pkl
import torch.nn as nn
import os.path as osp
import transformers as trs
import torch.utils.data as data
import torch.nn.functional as F

from sklearn import metrics
from collections import Counter
from collections import defaultdict

from io_utils import *
from load_data import *

# test loop / eval loop
def evaluate(dataloader, model, projection, args):

    # set model to eval mode
    model.eval()
    projection.eval()

    ranking_list = list()
    # iterate over dataset
    for batch in dataloader:
        # bs x dim
        context_text = batch[0].cuda()
        # bs x dim
        context_mask = batch[1].cuda()
        # bs x 10 x dim
        response_text = batch[2].cuda()
        # bs x 10 x dim
        response_mask = batch[3].cuda()
        # (bs * 10) x dim
        response_text = response_text.view(-1, response_text.size(2))
        # (bs * 10) x dim
        response_mask = response_mask.view(-1, response_mask.size(2))
        with torch.no_grad():

            # get features from last layer of bert
            # bs x dim
            context_features = model(input_ids=context_text, attention_mask=context_mask)[0]
            # (bs * 10) x dim
            response_features = model(input_ids=response_text, attention_mask=response_mask)[0]

            # take mean over all words
            context_features = torch.mean(context_features, 1)
            response_features = torch.mean(response_features, 1)

            # project the features
            # bx x 10 x dim
            response_features = response_features.view(-1, 10, response_features.size(1))
            context_projection = projection(context_features)
            # bs x 10 x dim
            response_projection = response_features


            # bs x 1 x dim
            context_projection = context_projection.unsqueeze(1).repeat(1, 10, 1)

            # take dot product for similarity
            similarities = torch.sum(context_projection * response_projection, dim=2)
            # bs x 10
            ranking = torch.argsort(similarities, dim=1, descending=True)

            # store all rankings
            ranking_list.append(ranking.detach().cpu())

    # compute various metrics
    rankings = torch.cat(ranking_list)
    
    return recall(rankings, topk=(1, 2, 3, 4, 5))

def recall(rankings, topk=(1,)):
    # code borrowed from official pytorch discussion forum
    with torch.no_grad():
        correct = rankings.eq(0)
        res = []
        for k in topk:
            correct_k = correct[:, :k].float().sum()
            res.append(correct_k.mul_(1.0 / rankings.size(0)).item())
    return res


if __name__ == '__main__':

    args = parse_args()

    # create tokenizer
    tokenizer = trs.BertTokenizer(vocab_file='bert_vocab.txt', do_lower_case=True)

    assert args.split in ['valid', 'test'], f"{args.split} not allowed"

    # create dataloader
    if args.dataset.startswith('udc'):
        dataset = UDC(root=args.dataset_root, split=args.split, tokenizer=tokenizer)
        dataloader = data.DataLoader(dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=2
        )
    else:
        raise Exception(F"unknown dataset: {args.dataset}")

    # check directory for experiment
    if args.exp_name is not None:
        args.checkpoint_dir = '%s/%s/%s' %(SAVE_DIR, args.dataset, args.exp_name)
    else:
        args.checkpoint_dir = '%s/%s/%s' %(SAVE_DIR, args.dataset, args.config_name)
    
    num_labels = 2
    
    # get default bert config
    config = trs.BertConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, 
        num_labels=num_labels, 
        finetuning_task="ir-chatbot"
    )

    # set concatenated vocab size
    config.vocab_size = 50155
    print('bert config')
    print(config)

    # make model
    model = trs.BertModel(config=config)
    # make projection layer
    projection =  nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                nn.LeakyReLU(),
                                nn.Linear(config.hidden_size, config.hidden_size),
                                nn.LeakyReLU(),
                                nn.Linear(config.hidden_size, config.hidden_size))

    # dataparallel model 
    model = nn.DataParallel(model.cuda())
    projection = nn.DataParallel(projection.cuda())

    print('checkpoints dir : ',args.checkpoint_dir)

    # resume file selection
    # file load selection
    if args.iter !=-1:
        resume_file = get_assigned_file(args.checkpoint_dir, args.iter)
    else:
        resume_file = get_resume_file(args.checkpoint_dir)

    assert osp.exists(resume_file), f"{resume_file} not found"

    print("trying to load from", resume_file)
    tmp = torch.load(resume_file)
    start_epoch = tmp['epoch'] + 1
    projection.load_state_dict(tmp['projection'])
    model.load_state_dict(tmp['feature'])

    # hack for loading dataparallel models
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(projection, 'module'):
        projection = projection.module

    # do eval
    results = evaluate(dataloader, model, projection, args)
    print('top-1,2,3,4,5 acc')
    print(results)