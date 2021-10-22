# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


# COMPUTE HINGE LOSS FOR CLAPP 
class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        
    def forward(self, input, positive):
        # positive IS USED TO KNOW THE POSITIVES IN THE SCORES MATRIX (IN OUR CASE, IT'S THE DIAGONAL)
        input[positive, positive] *= -1
        # APPLYING THE MAX(1+INPUT,0)
        input = torch.clamp(1+input, min=0)
        # SUMMATION OF THE 2 LOSS COMPONENTS 
        loss = 0.5*(torch.mean(input[positive, positive]) + (torch.sum(input)- torch.sum(input[positive, positive]))/((positive.size(0)-1)*(positive.size(0))))
        return loss


# COMPUTE THE TOP-K ACCURACIES
class top_k(nn.Module):
    def __init__(self, k):
        super(top_k, self).__init__()
        # PUT K IN FORM OF LIST IF K IS SINGLE INT VALUE
        k = [k] if isinstance(k, int) else k
        self.k=k
        
    def forward(self, input):
        accs = []
        #FIND TOP-K FOR EVERY K IN LIST
        for k in self.k:
            acc_k = []
            # FOR EACH TIMESTEP THAT WE WANT TO PREDICT
            for time_pred in input:
                # POSITIVE IS DIAGONAL, SPOT EVERY TIME THE DIAG ELEMENT IS IN THE TOP-K
               acc_k.append(torch.mean(torch.tensor([(index == input_line).any().float() for (index, input_line) in enumerate(torch.topk(time_pred, k, dim=1)[1])])))
              # AVERAGE THE SCORE OVER ALL TIME STEPS WE TRY TO PREDICT
            accs.append(torch.mean(torch.tensor(acc_k)))  
        return accs
            
        

class Loss(nn.Module):
    def __init__(self, mode, spatial_collapse, single_predictor, spatial_segm, predictor_bias, channels):
        super(Loss, self).__init__()
        # DEPENDING OF mode WE USE A DIFFERENT CATEGORICAL LOSS
        if mode =='CPC' or mode == 'GIM':
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = HingeLoss()
        
        # SETS THE USE OF ONE PREDICTOR APPLIED RECURSIVELY FOR ALL TIMESTEPS OR MULTIPLE PREDICTORS
        self.single_predictor = single_predictor
        # DEFINES IF THE SPATIAL MAPS ARE GOING TO BE SEGMENTED TO CREATE SPATIAL NEGATIVES (DPC TECHNIQUE)
        self.spatial_segm = spatial_segm
        # DEFINES IF THE ACTIVATION MAPS ARE POOLED TO FORM Z (ORIGINAL CPC TECHNIQUE, NOT USED HERE EVEN FOR CPC)
        self.spatial_collapse = spatial_collapse 
        # ADD A BIAS TO THE PREDICTION OPERATOR (NEVER USED)
        self.predictor_bias = predictor_bias
        
        if self.single_predictor:
            self.W = nn.Conv3d(channels, channels, kernel_size=(1,1,1), bias=self.predictor_bias)
        else:
            self.W = nn.ModuleList() 
            for i in range(3): # number of pred_steps
                self.W.append(nn.Conv3d(channels, channels, kernel_size=(1,1,1), bias=self.predictor_bias))
        

        # FOR FUTURE WORK: CREATION OF THE MASK IDENTIFYING THE TYPE OF SAMPLE: POS, TEMP. NEG., BATCH. NEG.
        self.mask_computed=False
        
        self._initialize_weights()
        
        
    def _initialize_weights(self):
        if self.single_predictor:
             nn.init.kaiming_normal_(self.W.weight, mode='fan_out', nonlinearity='relu')
             if self.W.bias is not None:
                 nn.init.constant_(self.W.bias, 0)
        else:
            for m in self.W:
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        
    def forward(self, input):
        
        if self.spatial_collapse:
            # IF SPATIAL COLLAPSE, PASS THROUGH POOLING
            collapse = nn.AvgPool3d(kernel_size=(1, input.size(-1), input.size(-1)))   
            input = collapse(input)
        
        # FIRST T-3 FRAMES TO USE FOR PREDICTION 
        state = input[:,:,:-3]
        # TARGET FOR PREDCTING 1 TIME STEP AHEAD
        first_targ = input[:,:,1:-2].permute(1,3,4,0,2)
        # TARGET FOR PREDCTING 2 TIME STEPS AHEAD
        second_targ = input[:,:,2:-1].permute(1,3,4,0,2)
        # TARGET FOR PREDCTING 3 TIME STEPS AHEAD
        third_targ = input[:,:,3:].permute(1,3,4,0,2)
        
        #PERMUTATION: (B,C,(T-3),X,Y) -> (C,X,Y,B,(T-3))
        if self.single_predictor:
            # APPLY PREDICTOR RECURSIVELY 
            first_pred = self.W(state).permute(1,3,4,0,2)
            second_pred = self.W(self.W(state)).permute(1,3,4,0,2)
            third_pred = self.W(self.W(self.W(state))).permute(1,3,4,0,2)

        
        else:
            first_pred = self.W[0](state).permute(1,3,4,0,2)
            second_pred =  self.W[1](state).permute(1,3,4,0,2)
            third_pred =  self.W[2](state).permute(1,3,4,0,2)
            
        # (C,X,Y,B,(T-3)) -> (CxXxY, Bx(T-3)) (FLATTENING ACTIVATION MAPS) or (C,XxYxBx(T-3)) (DPC)
        if self.spatial_segm:
            index = 1
        else:
            index = 3
            
        # PERFORM THE FLATTENING DEPENDING ON THE USE OF SPATIAL NEGATIVES
        first_targ = torch.flatten(torch.flatten(first_targ, start_dim=index), end_dim=index-1)
        second_targ =  torch.flatten(torch.flatten(second_targ, start_dim=index), end_dim=index-1)
        third_targ =  torch.flatten(torch.flatten(third_targ, start_dim=index), end_dim=index-1)
        
        first_pred =  torch.flatten(torch.flatten(first_pred, start_dim=index), end_dim=index-1)
        second_pred =  torch.flatten(torch.flatten(second_pred, start_dim=index), end_dim=index-1)
        third_pred =  torch.flatten(torch.flatten(third_pred, start_dim=index), end_dim=index-1)
            
        # COMPUTING THE SCORE BY MATRIX MULTIPLICATION  
        first_score = torch.matmul(first_targ.transpose(0,1),first_pred).transpose(0,1)
        second_score = torch.matmul(second_targ.transpose(0,1),second_pred).transpose(0,1)
        third_score = torch.matmul(third_targ.transpose(0,1),third_pred).transpose(0,1)
        
        # POSITIVE SAMPLES ARE THE DIAGONAL
        positive =  torch.arange(0,first_score.size(0)).cuda()
        
        return (self.loss(first_score, positive)+self.loss(second_score, positive)+self.loss(third_score, positive))/3, [first_score.detach(), second_score.detach(), third_score.detach()]
        
        
        
# CLASS FOR THE CONV+RELU+BN MODULE, NAME IS MISLEADING BUT BN IS LAST IN MODULE
class ConvBNReLU(nn.Module):
    def __init__(self,in_ch, out_ch, k_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        
        self.conv = nn.Conv3d(in_ch, out_ch, k_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        self.layers = nn.Sequential(self.conv, self.relu, self.norm)
            
        self._initialize_weights()
    
    def forward(self, x):
        return self.layers(x)

    def _initialize_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# CLASS FOR THE VGG-8 NETWORK
class VGG_8(nn.Module):
    def __init__(self, temp ,mode, spatial_collapse, single_predictor, spatial_segm, predictor_bias, no_ss_loss=False):
        super(VGG_8, self).__init__()
        
        self.mode = mode
        self.no_loss = no_ss_loss
        
        in_channels = 3
        
        #IF temp, SET THE LAST 2 CONVOLUTIONS WITH TEMPORAL KERNELS AND CONSEQUENT STRIDE
        if temp:
            time_kernel=3
        else:
            time_kernel=1
        
        self.conv1 = ConvBNReLU(in_channels, 96,(1, 7, 7), stride=(1,2,2), padding=0)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2))
        self.block1 = nn.Sequential(self.conv1, self.maxpool1)
        
        self.conv2 = ConvBNReLU(96, 256, (1,5,5), stride=(1,2,2), padding=(0,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2))
        self.block2 = nn.Sequential(self.conv2, self.maxpool2)

        
        self.conv3 = ConvBNReLU(256, 512, (1,3,3),stride=(1,1,1),padding=(0,1,1))
        self.block3 = nn.Sequential(self.conv3)
        self.conv4 = ConvBNReLU(512, 512, (time_kernel,3,3),stride=(time_kernel,1,1), padding=(0,1,1))
        self.block4 = nn.Sequential(self.conv4)
        self.conv5 = ConvBNReLU(512, 512, (time_kernel,3,3),stride=(time_kernel,1,1), padding=(0,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2))
        self.block5 = nn.Sequential(self.conv5, self.maxpool3)

        # LIST OF NETWORK BLOCKS, IF PER-LAYER, LOSS APPLIED TO EACH BLOCK
        self.blocks = nn.ModuleList([self.block1, self.block2 , self.block3, self.block4, self.block5])
        
        self.losses = nn.ModuleList()
        self.accs = nn.ModuleList()
        
        if self.mode =='GIM' or self.mode =='CLAPP':
            for block in self.blocks:
                self.losses.append(Loss(mode, spatial_collapse, single_predictor, spatial_segm, predictor_bias, block[0].conv.weight.size(0)))
                self.accs.append(top_k([1,3,5]))
        else:
            self.losses.append(Loss(mode, spatial_collapse, single_predictor, spatial_segm, predictor_bias, self.blocks[-1][0].conv.weight.size(0)))
            self.accs.append(top_k([1,3,5]))
            
    def get_nb_losses(self):
        return len(self.losses)
        
    
    def forward(self, input):
        self.losses_val = []
        self.accs_val = []

        res = input
        for i, block in enumerate(self.blocks):
            if self.mode =='CLAPP' or self.mode=='GIM':
                # IF CLAPP OR GIM, DISCONNECT INPUT 
                res = block(res.detach())
                if self.no_loss:
                    # OPTION TO PREVENT SELF-SUPERVISION LOSS IF CLASSIFICATION
                    loss = None
                    accs= None
                else:
                    # COMPUTE LOSS AND SCORES
                    loss, scores = self.losses[i](res)
                    accs = self.accs[i](scores)
                self.losses_val.append(loss)
                self.accs_val.append(accs)

            else:
                # IF NOT CLAPP OR GIM, FORWARD NORMALLY
                res = block(res)


        if self.mode =='CPC' or self.mode=='HingeCPC':
            # IF HINGECPC OR CPC, COMPUTE LOSS AND SCORES WITH LAST OUTPUT
            if self.no_loss:
                loss=None
                accs=None
            else:
                # INEX [0] BECAUSE LOSSES AND ACCS ARE ALWAYS A LIST
                loss, scores = self.losses[0](res)
                accs = self.accs[0](scores)
            self.losses_val.append(loss)
            self.accs_val.append(accs)
            
        return self.losses_val, self.accs_val, res
        
        
        
        