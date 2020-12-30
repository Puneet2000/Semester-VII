#!/usr/bin/env python3

__AUTHORS__ = [
    ('CS17BTECH11044', 'YASH KHASBAGE'),
    ('CS17BTECH11029', 'PUNEET MANGLA')
]

import os
import glob
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import os.path as osp
import torch.nn.functional as F

from nltk import tokenize
from collections import Counter
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from torch.nn.utils.weight_norm import WeightNorm
from sklearn.metrics import precision_recall_fscore_support
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from load_data import *

import warnings
warnings.filterwarnings("ignore")

# io_utils 
SAVE_DIR = '/DATA1/puneet/IR'

# parser
def parse_args():
    parser = argparse.ArgumentParser(
        description= 'IR Based Chatbot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", default="udc", type=str,
                        help="dataset")
    parser.add_argument("--exp_name", default=None, type=str,
                        help="dataset")
    parser.add_argument("--config_name", default='albert-base-v2', type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--split", default="train", type=str, 
                        choices=['train', 'test'],
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default='albert-base-v2', 
                        type=str)
    parser.add_argument("--dataset_root", type=str, default='./../', 
                        help='dataset root, should contain train.csv, test.csv, valid.csv inside it.')

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--resume", action='store_true',
                        help="Train using mixup")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
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
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    
    return parser.parse_args()

# get automated resume file
def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [x for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

# get resume file from details
def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

# train loop
def train(train_dataloader, model, projection, args):

    # very similar to main.py
    # loss
    criterion = torch.nn.BCELoss()

    # optimizer
    optimizer = AdamW([
            {'params': model.parameters()},
            {'params': projection.parameters()}
        ],                        
        lr=args.learning_rate, 
        eps=args.adam_epsilon
    )
    
    t_total = len(train_dataloader)*args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    
    # main loop
    for epoch in range(args.start_epoch,args.num_train_epochs):
        correct,total = 0,0
        avg_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            model.train()

            context_in    = batch[0].cuda()
            context_mask  = batch[1].cuda()
            response_in   = batch[2].cuda()
            response_mask = batch[3].cuda()
            label         = batch[4].cuda()

            # embeddings
            context_feature  = model(input_ids=context_in,  attention_mask=context_mask).last_hidden_state
            response_feature = model(input_ids=response_in, attention_mask=response_mask).last_hidden_state

            # mean aling words
            context_feature = torch.mean(context_feature, 1)
            response_feature = torch.mean(response_feature, 1)

            # projection and dot product
            logit = F.sigmoid(torch.sum(projection(context_feature) * response_feature, 1))

            # acc
            pred =  (logit > 0.5).long()
            correct += (pred == label).sum().item()
            total += pred.size(0)

            # loss
            loss = criterion(logit, label.float())
            loss = loss.mean()

            # back propagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.data.item()
            scheduler.step()

            # log
            if i % 10 == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:3f} | Accuracy {:3f}'.format(epoch, i, len(train_dataloader), \
                    avg_loss/(i+1), (100.*correct)/total))

        # save
        if (epoch % args.save_freq == 0) or (epoch == args.num_train_epochs - 1):
            outfile = os.path.join(args.checkpoint_dir, '{:d}.tar'.format(epoch))
            state_dict = {}
            state_dict['epoch'] = epoch
            state_dict['feature'] = model.state_dict()
            state_dict['projection'] =  projection.state_dict()
            torch.save(state_dict, outfile)


def evaluate(dataloader, model, projection, args):
    # test / eval loop
    
    # same as eval.py
    model.eval()
    projection.eval()

    ranking_list = list()
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
        row = batch[4]
        with torch.no_grad():
            # bs x dim
            context_features = model(input_ids=context_text, attention_mask=context_mask)[0]
            # (bs * 10) x dim
            response_features = model(input_ids=response_text, attention_mask=response_mask)[0]

            context_features = torch.mean(context_features,1)
            response_features = torch.mean(response_features,1)

            # bx x 10 x dim
            response_features = response_features.view(-1, 10, response_features.size(1))
            context_projection = projection(context_features)
            # bs x 10 x dim
            response_projection = response_features

            # bs x 1 x dim
            context_projection = context_projection.unsqueeze(1).repeat(1,10,1,)
            similarities = torch.sum(context_projection * response_projection, dim=2)
            # bs x 10
            ranking = torch.argsort(similarities, dim=1, descending=True)
            # print(ranking)

            ranking_list.append(ranking.detach().cpu())

    rankings = torch.cat(ranking_list)
    
    return recall(rankings, topk=(1, 2, 3, 4, 5))

# metric utils
def recall(rankings, topk=(1,)):

    with torch.no_grad():
        correct = rankings.eq(0)
        res = []
        for k in topk:
            correct_k = correct[:, :k].float().sum()
            res.append(correct_k.mul_(1.0 / rankings.size(0)).item())
    return res


if __name__ == '__main__':
    
    args = parse_args()

    if args.pdb:
        import pdb; pdb.set_trace()
    
    torch.cuda.set_device(args.cuda)

    # create tokenizer
    tokenizer = AlbertTokenizer.from_pretrained(args.config_name)
    
    # get data loader
    if args.dataset.startswith('udc'):
        dataset = UDC(root=args.dataset_root, split=args.split, tokenizer=tokenizer)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        num_labels = 2
    else:
        raise Exception("Unknown dataset: {}".format(args.dataset))

    config = AlbertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072
    )
    print('albert configs:')
    print(config)

    # get projection layer
    # same as main.py
    projection =  nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                nn.LeakyReLU(),
                                nn.Linear(config.hidden_size, config.hidden_size),
                                nn.LeakyReLU(),
                                nn.Linear(config.hidden_size, config.hidden_size))

    # make model
    model = AlbertModel.from_pretrained(args.config_name, return_dict=True)

    model.cuda()
    projection.cuda()

    # for fine tuning choices
    # for params in model.parameters():
    #     params.requires_grad = False

    # get paths for model weights store/load
    if args.exp_name is not None:
        args.checkpoint_dir = '%s/%s/%s' %(SAVE_DIR, args.dataset, args.exp_name)
    else:
        args.checkpoint_dir = '%s/%s/%s' %(SAVE_DIR, args.dataset, args.config_name)


    print('checkpoints dir : ',args.checkpoint_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    args.start_epoch = 0

    if args.resume:
        # for testing, one has to load models from certain path
        if args.iter !=-1:
            resume_file = get_assigned_file(args.checkpoint_dir, args.iter)
        else:
            resume_file = get_resume_file(args.checkpoint_dir)
        print('Resuming ', resume_file)
        if resume_file is not None:
            print('Resume file is: ', resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            projection.load_state_dict(tmp['projection'])
            model.load_state_dict(tmp['feature'])
        else:
            raise Exception('Resume file not found')

    if args.do_train:
        train(data_loader, model, projection, args)
    elif args.do_eval:
        results = evaluate(data_loader, model, projection, args)
        print('top-1,2,3,4,5 acc')
        print(results)