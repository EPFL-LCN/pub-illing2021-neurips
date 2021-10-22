# -*- coding: utf-8 -*-


from VGG_8 import VGG_8
from augmentation import *
from train import train, validate
from dataset_3d import *
from torch.utils import data

from tensorboardX import SummaryWriter
import os
import time
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', default=8, type=int, help='number of frames in each sequence')
parser.add_argument('--temp_VGG', action='store_true', help='standard or temporal VGG-8')
parser.add_argument('--mode', default='CPC', help='Self-supervised algorithm')
parser.add_argument('--spatial_collapse', action='store_true', help='performing average pooling or not to obtain z')
parser.add_argument('--spatial_segm', action='store_true', help='use of spatial negatives (if not, then flattening)')
parser.add_argument('--single_predictor', action='store_true', help='use of a single recursively applied predictor')
parser.add_argument('--predictor_bias', action='store_true', help='linear predicting layer having bias or not')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--name', help='relative path to store the model and the tensorboard files')

def run():    
    torch.autograd.set_detect_anomaly(True)

    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')
    
    img_path = args.name

    # CREATION OF THE NETWORK AND ITS LOSS
    model = VGG_8(args.temp_VGG, args.mode, args.spatial_collapse, args.single_predictor, args.spatial_segm, args.predictor_bias) 

    model = model.to(cuda)


    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)


    # DUMMY VALIDATION LOSS FOR SAVING BEST MODEL 
    best_loss = 100
    global iteration; iteration = 0


    # TRANSFORMATION FOR SELF-SUPERVISED TRAINING
    # ARGUMENT consistent SETS IF PROCESSING PER FRAME OR PER SEQUENCE
    transform = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        RandomCrop(size=224, consistent=True),
        Scale(size=(args.img_dim,args.img_dim)),
        RandomGray(consistent=False, p=0.5),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0, consistent=False),
        ToTensor(),
        Normalize()
    ])


    # CREATION OF DATALOADERS
    train_loader = get_data(transform, 'train')
    val_loader = get_data(transform, 'val')


    # INSTANTIATION OF THE TENSORBOARD MONITORING
    try: # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except: # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))
        
        
    for epoch in range(args.epochs):
        
        train_losses, train_accs = train(model, train_loader, optimizer, epoch)
        val_losses, val_accs = validate(model, val_loader, epoch)


        # SAVE CURVES, ITERATE OVER LOSSES OF THE NETWORK (1 LOSS IF END-TO-END AND N IF PER-LAYER)
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            writer_train.add_scalar('global/loss_{}'.format(i), train_loss, epoch)
            writer_val.add_scalar('global/loss_{}'.format(i), val_loss, epoch)
        
        
        # SAVE CURVES, ITERATE OVER ACCURACIES OF THE NETWORK ([3] ACCURACIES IF END-TO-END AND N*[3] IF PER-LAYER)
        for i, (train_acc, val_acc) in enumerate(zip(train_accs, val_accs)):
            # EACH LOSS IS ASSOCIATED WITH TOP-1,3,5 ACCURACIES  
            for j in range(3):
                a= [1,3,5]
                writer_train.add_scalar('global/accuracy_{}_top_{}'.format(i,a[j]), train_acc[j], epoch)
                writer_val.add_scalar('global/accuracy_{}_top_{}'.format(i, a[j]), val_acc[j], epoch)
               
                
        # SAVE MODEL IF BEST VALIDATION LOSS
        if val_losses[-1] <= best_loss:
            best_loss = val_losses[-1]
            torch.save(model.state_dict(), img_path+'/model.pth.tar')
            
        print('epoch {}/{}'.format(epoch, args.epochs))
        
        
        
def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)

    dataset = UCF101_3d(mode=mode,
                     transform=transform,
                     seq_len=args.seq_len,
                     num_seq=1, # NUMBER OF SEQUENCES, ARTEFACT FROM DPC CODE, KEEP SET TO 1! 
                     downsample=3) # FRAME RATE DOWNSAMPLING: FPS = 30/downsample
    sampler = data.RandomSampler(dataset)

    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=True)

    print('"%s" dataset size: %d' % (mode, len(dataset)))
    
    return data_loader
    
if __name__ == '__main__':
    args = sys.argv

    run()