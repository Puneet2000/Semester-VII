#!/usr/bin/env python3
__AUTHORS__ = [
    ('PUNEET MANGLA', 'CS17BTECH11029'),
    ('YASH KHASBAGE', 'CS17BTECH11044')
]

import os
import torch
import pickle
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from collections import Counter
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from io_utils import *
from load_data import *


# train loop
def train(train_dataloader, model, projection, args):
    # Prepare optimizer and schedule (linear warmup and decay)
    
    # loss function: binary cross entropy
    criterion = torch.nn.BCELoss()

    # optimizer AdamW
    optimizer = AdamW([
            {'params': model.parameters()},
            {'params': projection.parameters()},
        ],
        lr=args.learning_rate, 
        eps=args.adam_epsilon
    )

    t_total = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    
    # iterate over epochs
    for epoch in range(args.start_epoch,args.num_train_epochs):
        correct, total = 0, 0
        avg_loss = 0.0

        # iterate over train set
        for i, batch in enumerate(train_dataloader):

            # set model to train mode
            model.train()

            context_in    = batch[0].cuda()
            context_mask  = batch[1].cuda()
            response_in   = batch[2].cuda()
            response_mask = batch[3].cuda()
            label         = batch[4].cuda()

            # get bert embeddings from last layer
            context_feature   = model(input_ids=context_in,  attention_mask=context_mask)[0]
            response_feature  = model(input_ids=response_in, attention_mask=response_mask)[0]
            
            # get mean over all words in text
            context_feature  = torch.mean(context_feature,  1)
            response_feature = torch.mean(response_feature, 1)

            # get projection of context feature
            context_projection = projection(context_feature)
            response_projection = response_feature

            # dot product of context and response
            logit = F.sigmoid(torch.sum(context_projection * response_projection, 1))

            # compute correct predictions in batch
            pred =  (logit > 0.5).long()
            correct += (pred == label.cuda()).sum().item()
            total += pred.size(0)

            # compuet loss by comparing labels and logits
            loss = criterion(logit, label.float())
            loss = loss.mean()

            # propagate loss and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute average loss and do scheduler step
            avg_loss += loss.item()
            scheduler.step()

            if i % 10 == 0:
                # logging
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:3f} | Accuracy {:3f}'.format(epoch, i, len(train_dataloader), \
                    avg_loss/(i+1), (100.*correct)/total))

        # save models after some intervals
        if (epoch % args.save_freq == 0) or (epoch == args.num_train_epochs - 1):
            outfile = os.path.join(args.checkpoint_dir, '{:d}.tar'.format(epoch))
            state_dict = {}
            state_dict['epoch'] = epoch
            state_dict['feature'] = model.state_dict()
            state_dict['projection'] =  projection.state_dict()
            torch.save(state_dict, outfile)


if __name__ == '__main__':
    
    args = parse_args()
    
    # create tokenizer
    tokenizer = BertTokenizer(vocab_file='bert_vocab.txt', do_lower_case=True)

    # get data loader
    if args.dataset.startswith('udc'):
        dataset = UDC(root=args.dataset_root, split=args.split, tokenizer=tokenizer)
        data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        num_labels = 2
    else:
        raise Exception("Unknown dataset: {}".format(args.dataset))

    # get bert config
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, 
                                        num_labels=num_labels, finetuning_task="ir")
    
    # concatenated vocabulary
    config.vocab_size = 50155
    print('bert configs:')
    print(config)

    # projection layer
    # for aligning context and response features
    projection =  nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                nn.LeakyReLU(),
                                nn.Linear(config.hidden_size, config.hidden_size),
                                nn.LeakyReLU(),
                                nn.Linear(config.hidden_size, config.hidden_size))
    

    # get bert(pretrained)
    model = BertModel(config=config)

    model.cuda()
    projection.cuda()

    model = nn.DataParallel(model)
    projection = nn.DataParallel(projection)

    # for fine tuning only projection layer
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
        if resume_file is not None:
            print('Resume file is: ', resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            projection.load_state_dict(tmp['projection'])
            model.load_state_dict(tmp['feature'])
        else:
            raise Exception('Resume file not found')

    train(data_loader, model, projection, args)

