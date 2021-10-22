# -*- coding: utf-8 -*-
from VGG_8 import VGG_8
from augmentation import *
from dataset_3d_lc import *
from torch.utils import data
from tqdm import tqdm
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
from utils import AverageMeter

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', default=8, type=int, help='number of frames in each sequence')
parser.add_argument('--temp_VGG', action='store_true', help='standard or temporal VGG-8')
parser.add_argument('--mode', default='CPC', help='Self-supervised algorithm, necessary for retrieving saved network structure')
parser.add_argument('--spatial_collapse', action='store_true', help='performing average pooling or not to obtain z')
parser.add_argument('--spatial_segm', action='store_true', help='use of spatial negatives (if not, then flattening)')
parser.add_argument('--single_predictor', action='store_true', help='use of a single recursively applied predictor')
parser.add_argument('--predictor_bias', action='store_true', help='linear predicting layer having bias or not')
parser.add_argument('--monitor_all_layers', action='store_true', help='perform the classification at each layer')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--name', help='relative path to load trained encoder and store the model and the tensorboard files')


# CLASS PERFORMING TOP-K ACCURACY FOR CLASSIFICATION
class top_k(nn.Module):
    def __init__(self, k):
        super(top_k, self).__init__()
        k = [k] if isinstance(k, int) else k
        self.k=k
        
    def forward(self, input, targets):
        accs = []
        for k in self.k:
            acc_k = torch.mean(torch.tensor([(target == input_line).any().float() for (target, input_line) in zip(targets,torch.topk(input, k, dim=1)[1])]))
            accs.append(acc_k)
        return accs
    

def classify():    
    torch.autograd.set_detect_anomaly(True)

    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')
    
    img_path = args.name
    # CREATING ENCODER MODEL, LAST ARGUMENT set as True PREVENTS COSTLY COMPUTATION OF SELF-SUPERVISED LOSSES AND ACCS  
    base_model = VGG_8(args.temp_VGG, args.mode, args.spatial_collapse, args.single_predictor, args.spatial_segm, args.predictor_bias, True) 
    # IF MODEL FOUND IN FOLDER DESIGNATED BY name, LOAD PARAMETERS
    if os.path.isfile(img_path+'/model.pth.tar'):
                 base_model.load_state_dict(torch.load(img_path+'/model.pth.tar'))
    else:
        print('file not found, starts with random encoder')
        

    # FAKE INPUT TO COMPUTE SIZE OF CLASSIFIERS (AT EACH LAYER OR JUST AT THE END)
    input = torch.randn(1,3,args.seq_len,args.img_dim,args.img_dim)
    output_sizes = []
    for block in base_model.blocks:
        input = block(input.detach())
        if args.monitor_all_layers:
            # WE DO NOT COUNT TIME DIMENSION (2) BECAUSE IT IS AVERAGE POOLED
            output_sizes.append([int(torch.numel(input)/input.size(2)), input.size(2)])

    if not args.monitor_all_layers:
        output_sizes.append([int(torch.numel(input)/input.size(2)),input.size(2)])
        
        
    # CREATION OF THE CLASSIFIER(S) FOR EACH OUTPUT SIZE
    classifications = nn.ModuleList()
    for i, output_size in enumerate(output_sizes):
        classifications.append(nn.Sequential(nn.AvgPool3d((output_size[1],1,1)),nn.Flatten(),nn.BatchNorm1d(output_size[0]), nn.Dropout(0.5), nn.Linear(output_size[0], 101)))
        classifications[i][2].weight.data.fill_(1)
        classifications[i][2].bias.data.zero_()    
    
        for name, param in classifications[i][-1].named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)  
    
    
    # MIGRATING MODEL AND CLASSIFIERS TO CUDA
    base_model = base_model.to(cuda)
    classifications = classifications.to(cuda)


    # SETTING THE FINETUNING, CLASSIFIER WITH lr AND ENCODER WITH lr/10
    print('=> finetune backbone with smaller lr')
    params = []
    for name, param in base_model.named_parameters():
        params.append({'params': param, 'lr': args.lr/10})
    for name, param in classifications.named_parameters():
        params.append({'params': param})


    # CHECKING GRADIENTS OF DIFFERENT COMPONENTS
    print('\n===========Check Grad============')
    for name, param in base_model.named_parameters():
        print(name, param.requires_grad)
    for name, param in classifications.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')


    # GIVE THE PARAMETERS TO THE OPTIMIZER
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=[60, 80, 100], repeat=1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    # DUMMY VALIDATION LOSS FOR SAVING BEST MODEL 
    best_loss = 100
    global iteration; iteration = 0


    # DEFINE THE TRANSFORMATIONS FOR TRAIN AND VALIDATION
    transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=1.0),
        Scale(size=(args.img_dim,args.img_dim)),
        RandomHorizontalFlip(consistent=True),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
    ])
    val_transform = transforms.Compose([
        RandomSizedCrop(consistent=True, size=224, p=0.3),
        Scale(size=(args.img_dim,args.img_dim)),
        RandomHorizontalFlip(consistent=True),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3, consistent=True),
        ToTensor(),
        Normalize()
    ])

    train_loader = get_data(transform, 'train')
    val_loader = get_data(val_transform, 'val')
    
    appendix = '_finetune'
        
    
    # INSTANTIATION OF THE TENSORBOARD MONITORING
    try: # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'classification'+appendix+'/val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'classification'+appendix+'/train'))
    except: # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'classification'+appendix+'/val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'classification'+appendix+'/train'))
        
        
    for epoch in range(args.epochs):
        train_losses, train_accs = train(base_model, classifications, train_loader, optimizer, epoch, args.monitor_all_layers)
        val_losses, val_accs = validate(base_model, classifications, val_loader, epoch, args.monitor_all_layers)

        scheduler.step()


        # SAVE CURVES, ITERATE OVER LOSSES OF THE NETWORK (1 LOSS IF END-TO-END AND N IF PER-LAYER)        
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            writer_train.add_scalar('global/loss_{}'.format(i), train_loss, epoch)
            writer_val.add_scalar('global/loss_{}'.format(i), val_loss, epoch)
            
            
        # SAVE CURVES, ITERATE OVER ACCURACIES OF THE NETWORK ([3] ACCURACIES IF END-TO-END AND N*[3] IF PER-LAYER)
        for i, (train_acc, val_acc) in enumerate(zip(train_accs, val_accs)):
            for j in range(3):
                a= [1,3,5]
                writer_train.add_scalar('global/accuracy_{}_top_{}'.format(i,a[j]), train_acc[j], epoch)
                writer_val.add_scalar('global/accuracy_{}_top_{}'.format(i, a[j]), val_acc[j], epoch)
      
                
        # SAVE MODEL IF BEST VALIDATION LOSS
        if val_losses[-1] <= best_loss:
            best_loss = val_loss
            torch.save(classifications.state_dict(), img_path+'/classifier'+appendix+'.pth.tar')
            
        print('epoch {}/{}'.format(epoch, args.epochs))
        
        
        
def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    global dataset
    dataset = UCF101_3d(mode=mode, 
                     transform=transform, 
                     seq_len=args.seq_len,
                     num_seq=1,  # NUMBER OF SEQUENCES, ARTEFACT FROM DPC CODE, KEEP SET TO 1!
                     downsample=3) # FRAME RATE DOWNSAMPLING: FPS = 30/downsample

    my_sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=16,
                                      pin_memory=True,
                                      drop_last=True)

    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader



def train(model, classifiers, data_loader, optimizer, epoch, monitor_all_layers):
    cuda = torch.device('cuda')
    # SET THE LOSSES AND ACCURACIES
    # WARNING: USING x*[OBJECT] DUPLICATES REFERENCES TO THE SAME OBJECT INSTANCE
    #          HENCE THE FOR LOOP
    losses = []
    accuracies = []
    Losses = [] 
    Accs = [] 
    if isinstance(classifiers, nn.ModuleList):
        for i in range(len(classifiers)):
            losses.append(AverageMeter())
            accuracies.append([AverageMeter(),AverageMeter(),AverageMeter()])
            Losses.append(nn.CrossEntropyLoss())
            Accs.append( top_k([1,3,5]))
            
    model.train()
    classifiers.train()
    
    for (input_seq, target) in tqdm(data_loader):
        # CREATING THE LIST OF NETWORK OUTPUTS AND CLASSIFICATION LOSSES
        res_losses = []
        outputs = []

        input_seq = input_seq.squeeze().to(cuda)
        target = target.squeeze().to(cuda)
        B = input_seq.size(0)
        
        
        # IF ONLY CLASSIFICATION AT FINAL LAYER
        if not monitor_all_layers:
            _, _, output = model(input_seq)
            outputs.append(output)
        else:
            output=input_seq

            for block in model.blocks:
                # OTHERWISE AT EACH LAYER 
                output = block(output)
                outputs.append(output)


        # MEASURE THE CLASSIFICATION PERFORMANCE 
        for output, classifier, Loss, Acc, loss, accuracy in zip(outputs, classifiers, Losses, Accs, losses, accuracies):
            # PASS THE OUTPUT(S) TO ITS/THEIR CLASSIFIER 
            output = classifier(output)
            # COMPUTE THE CLASSIFIER'S LOSS AND ACCURACIES
            l = Loss(output, target)
            res_losses.append(l)
            acc = Acc(output, target)
            loss.update(l.item(), B)
            for j in range(3):
                accuracy[j].update(acc[j].item(), B)


        # BACKWARD AND UPDATE THE LOSS RESULTING FROM THE LAST OUTPUT
        optimizer.zero_grad()
        res_losses[-1].backward()
        optimizer.step()
        
        
    # PRINT PERFORMANCES INDEXES AT EVERY EPOCH
    for loss, acc in zip(losses, accuracies):
        print('Training loss: {:.4f} | top1: {:.4f}  | top3: {:.4f} | top5: {:.4f}'.format(loss.avg, acc[0].avg, acc[1].avg ,acc[2].avg))
    return [loss.local_avg for loss in losses], [[acc[0].avg,acc[1].avg,acc[2].avg] for acc in accuracies]
    


def validate(model, classifiers, data_loader, epoch, monitor_all_layers):
    cuda = torch.device('cuda')
    # SET THE LOSSES AND ACCURACIES
    # WARNING: USING x*[OBJECT] DUPLICATES REFERENCES TO THE SAME OBJECT INSTANCE
    #          HENCE THE FOR LOOP
    losses = []
    accuracies = []
    Losses = [] 
    Accs = [] 
    if isinstance(classifiers, nn.ModuleList):
        for i in range(len(classifiers)):
            losses.append(AverageMeter())
            accuracies.append([AverageMeter(),AverageMeter(),AverageMeter()])
            Losses.append(nn.CrossEntropyLoss())
            Accs.append( top_k([1,3,5]))
            
    model.eval()
    classifiers.eval()

    for (input_seq, target) in tqdm(data_loader):
        # CREATING THE LIST OF NETWORK OUTPUTS AND CLASSIFICATION LOSSES
        outputs = []
        input_seq = input_seq.squeeze().to(cuda)
        target = target.squeeze().to(cuda)
        B = input_seq.size(0)
        

        # IF ONLY CLASSIFICATION AT FINAL LAYER
        if not monitor_all_layers:
            _, _, output = model(input_seq)
            outputs.append(output)
        else:
            output=input_seq

            for block in model.blocks:
                # OTHERWISE AT EACH LAYER 
                output = block(output)
                outputs.append(output)
                
            
        # MEASURE THE CLASSIFICATION PERFORMANCE 
        for output, classifier, Loss, Acc, loss, accuracy in zip(outputs, classifiers, Losses, Accs, losses, accuracies):
            # PASS THE OUTPUT(S) TO ITS/THEIR CLASSIFIER 
            output = classifier(output.detach())
            # COMPUTE THE CLASSIFIER'S LOSS AND ACCURACIES
            l = Loss(output, target)
            acc = Acc(output, target)

            loss.update(l.item(), B)
            for j in range(3):
                accuracy[j].update(acc[j].item(), B)
        
        
    # PRINT PERFORMANCES INDEXES AT EVERY EPOCH
    for loss, acc in zip(losses, accuracies):
        print('Validation loss: {:.4f} | top1: {:.4f}  | top3: {:.4f} | top5: {:.4f}'.format(loss.avg, acc[0].avg, acc[1].avg ,acc[2].avg))
    return [loss.local_avg for loss in losses], [[acc[0].avg,acc[1].avg,acc[2].avg] for acc in accuracies]



# USE OF THE SAME LEARNING RATE SCHEDULER AS DPC, SHOULD BE TAKEN AWAY FOR MORE STABLE RESULTS
def MultiStepLR_Restart_Multiplier(epoch, gamma=0.1, step=[10,15,20], repeat=3):
    '''return the multipier for LambdaLR, 
    0  <= ep < 10: gamma^0
    10 <= ep < 15: gamma^1 
    15 <= ep < 20: gamma^2
    20 <= ep < 30: gamma^0 ... repeat 3 cycles and then keep gamma^2'''
    max_step = max(step)
    effective_epoch = epoch % max_step
    if epoch // max_step >= repeat:
        exp = len(step) - 1
    else:
        exp = len([i for i in step if effective_epoch>=i])
    return gamma ** exp


if __name__ == '__main__':
    args = sys.argv

    classify()