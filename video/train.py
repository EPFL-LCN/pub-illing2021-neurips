# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
from utils import AverageMeter


def train(model, data_loader, optimizer, epoch):
    cuda = torch.device('cuda')
    # SET THE LOSSES AND ACCURACIES
    # WARNING: USING x*[OBJECT] DUPLICATES REFERENCES TO THE SAME OBJECT INSTANCE
    #          HENCE THE FOR LOOP
    losses = []
    accuracies = []
    for i in range( model.get_nb_losses()):
        losses.append(AverageMeter())
        accuracies.append([AverageMeter(),AverageMeter(),AverageMeter()])
        
    model.train()
    
    for input_seq in tqdm(data_loader):
        # INPUT HAS DIMENSION (B, 1, 3, T, X, Y)
        # 1 COMES FROM DPC TREATING SEQUENCES OF SEQUENCES,SOLVED WITH SQUEEZE
        input_seq = input_seq.squeeze().to(cuda)
        B = input_seq.size(0)
        
        
        # MODEL OUTPUTS LOSSES, ACCURACIES, Z
        res_losses, res_accs, _ = model(input_seq)


        # UPDATE THE RUNNING LOSSES AND ACCURACIES
        for i,(res_loss, res_acc) in enumerate(zip(res_losses, res_accs)):
            losses[i].update(res_loss.item(), B)
            for j in range(3):
                accuracies[i][j].update(res_acc[j].item(), B)


        # PERFORM BACKWARD(S) AND BACK-PROPAGATION
        optimizer.zero_grad()
        for loss in res_losses:
            loss.backward()
        optimizer.step()
        
        
    # PRINT PERFORMANCES INDEXES AT EVERY EPOCH
    for loss, acc in zip(losses, accuracies):
        print('Training loss: {:.4f} | top1: {:.4f}  | top3: {:.4f} | top5: {:.4f}'.format(loss.avg, acc[0].avg, acc[1].avg ,acc[2].avg))
    return [loss.local_avg for loss in losses], [[acc[0].avg,acc[1].avg,acc[2].avg] for acc in accuracies]



def validate(model, data_loader, epoch):
    cuda = torch.device('cuda')
    # SET THE LOSSES AND ACCURACIES
    # WARNING: USING x*[OBJECT] DUPLICATES REFERENCES TO THE SAME OBJECT INSTANCE
    #          HENCE THE FOR LOOP   
    losses = []
    accuracies = []
    for i in range(model.get_nb_losses()):
        losses.append(AverageMeter())
        accuracies.append([AverageMeter(),AverageMeter(),AverageMeter()])
        
    model.eval()
    
    for input_seq in tqdm(data_loader):
        # INPUT HAS DIMENSION (B, 1, 3, T, X, Y)
        # 1 COMES FROM DPC TREATING SEQUENCES OF SEQUENCES,SOLVED WITH SQUEEZE
        input_seq =  input_seq.squeeze().to(cuda)
        B = input_seq.size(0)
        
        
        # MODEL OUTPUTS LOSSES, ACCURACIES, Z
        res_losses, res_accs, _ = model(input_seq)
        
        
        # UPDATE THE RUNNING LOSSES AND ACCURACIES
        for i,(res_loss, res_acc) in enumerate(zip(res_losses, res_accs)):
            losses[i].update(res_loss.item(), B)
            for j in range(3):
                accuracies[i][j].update(res_acc[j].item(), B)
                
                
    # PRINT PERFORMANCES INDEXES AT EVERY EPOCH
    for loss, acc in zip(losses, accuracies):
        print('Validation loss: {:.4f} | top1: {:.4f}  | top3: {:.4f} | top5: {:.4f}'.format(loss.avg, acc[0].avg, acc[1].avg ,acc[2].avg))
    return [loss.local_avg for loss in losses], [[acc[0].avg,acc[1].avg,acc[2].avg] for acc in accuracies]