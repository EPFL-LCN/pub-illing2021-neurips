# Code to compare numerically the updates stemming from (i) CLAPP learning rules and (ii) CLAPP loss + autodiff in pytorch
# ATTENTION: This is tested for 
# 1) using the same (single) negative everywhere (local sampling): --sample_negs_locally --sample_negs_locally_same_everywhere
# 2) not using W_retro for the moment (i.e. NOT --asymmetric_W_pred)

# Respective simulations need to be run/created (running 'CLAPPVision.vision.main_vision' with the same command line options.
# However the below tests should hold at any point of training, e.g. also at the first epoch of training

# Bash command for tested cases:
# python -m CLAPPVision.vision.compare_updates --download_dataset --save_dir CLAPP_1 --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --contrast_mode 'hinge' --num_epochs 600 --negative_samples 1 --sample_negs_locally --sample_negs_locally_same_everywhere --start_epoch 598 --model_path ./logs/CLAPP_1/ --save_vars_for_update_calc 3 --batch_size 4
# python -m CLAPPVision.vision.compare_updates --download_dataset --save_dir CLAPP_2 --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --contrast_mode 'hinge' --num_epochs 600 --negative_samples 1 --sample_negs_locally --sample_negs_locally_same_everywhere --either_pos_or_neg_update --start_epoch 599 --model_path ./logs/CLAPP_2/ --save_vars_for_update_calc 3 --batch_size 4

################################################################################

from shutil import which
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import choice
import time
import os
import code
import sklearn
from IPython import embed
import matplotlib.pyplot as plt
import copy

## own modules
from CLAPPVision.vision.data import get_dataloader
from CLAPPVision.vision.arg_parser import arg_parser
from CLAPPVision.vision.models import load_vision_model
from CLAPPVision.utils import logger, utils


def train_iter(opt, model, train_loader):
    model.module.switch_calc_loss(True)

    starttime = time.time()
    cur_train_module = opt.train_module
    
    img, label = next(iter(train_loader))
    
    model_input = img.to(opt.device)
    label = label.to(opt.device)
    loss, loss_gated, _, z, accuracy = model(model_input, label, n=cur_train_module)
    loss = torch.mean(loss, 0) # take mean over outputs of different GPUs

    model.zero_grad()
    for idx in range(len(loss)):
        if idx == len(loss) - 1: # last module
            loss[idx].backward()
        else:
            # select loss or loss_gated for gradient descent
            loss_for_grads = loss_gated[idx] if model.module.opt.feedback_gating else loss[idx]
            loss_for_grads.backward(retain_graph=True)


def _load_activations(opt, layer, k):
    which_update = torch.load(os.path.join(opt.model_path, 'saved_which_update_layer_'+str(layer)), map_location=torch.device('cpu'))
    
    (context, z_p, z_n, rand_index) = torch.load(os.path.join(opt.model_path, 'saved_c_and_z_layer_'+str(layer)+'_k_'+str(k)), map_location=torch.device('cpu'))
    context = context.squeeze(-2)
    # all size: y (red.), x, b, c

    (loss_p, loss_n) = torch.load(os.path.join(opt.model_path, 'saved_losses_layer_'+str(layer)+'_k_'+str(k)), map_location=torch.device('cpu'))
    # b, 1, y (red.), x
    dloss_p = - torch.sign(loss_p.squeeze(1).permute(1, 2, 0))
    dloss_n = torch.sign(loss_n.squeeze(1).permute(1, 2, 0))
    # y (red.), x, b

    return which_update, context, z_p, z_n, rand_index, dloss_p, dloss_n

def _add_pos_and_neg_dW(which_update, dW_p, dW_n):
    def _select_dWs(dW, inds):
        if sum(inds) > 0: # exclude empty sets which lead no NaN in loss
            if len(dW.shape) == 5:
                dW = dW[:, :, inds, :, :]
            elif len(dW.shape) == 7:
                dW = dW[:, :, inds, :, :, :, :]
        else:
            dW[:] = 0.
        return dW

    if type(which_update) != str:
        inds_p = torch.tensor((which_update == 'pos').tolist())
        inds_n = torch.tensor((which_update == 'neg').tolist())
        dW_p = _select_dWs(dW_p, inds_p)
        dW_n = _select_dWs(dW_n, inds_n)

    dW_p_m = dW_p.mean(dim=(0,1,2)) # mean over y (red.), x, b
    dW_n_m = dW_n.mean(dim=(0,1,2))
    
    dW = dW_p_m + dW_n_m
    return dW

def _get_dWpred(opt, layer, k):
    which_update, context, z_p, z_n, _, dloss_p, dloss_n = _load_activations(opt, layer, k)

    # post * pre
    dWpred_p = torch.einsum("yxbc, yxbd -> yxbcd", context, z_p)
    dWpred_n = torch.einsum("yxbc, yxbd -> yxbcd", context, z_n)
    # n_p_y (red.), n_p_x, b, c, c
    
    # * "gamma"
    dWpred_p = dloss_p.unsqueeze(-1).unsqueeze(-1) * dWpred_p
    dWpred_n = dloss_n.unsqueeze(-1).unsqueeze(-1) * dWpred_n

    dWpred = _add_pos_and_neg_dW(which_update, dWpred_p, dWpred_n) # c, c
    
    dWpred /= opt.prediction_step # this factor appears in loss

    return dWpred

def compare_Wpred(opt, model, k):
    layer = opt.save_vars_for_update_calc
    
    # update acc. to CLAPP rule 
    dWpred = _get_dWpred(opt, layer, k)

    grad_Wpred = model.module.model[0][layer-1].loss.W_k[k-1].weight.grad.squeeze().clone().detach().to('cpu')
    # model.module.model[0][layer-1].loss_mirror.W_k[0].weight.grad

    diff = dWpred - grad_Wpred
    d = diff.norm() / (dWpred.norm() + grad_Wpred.norm())

    return diff, d, grad_Wpred, dWpred


def _get_dW_ff(opt, layer, skip_step=1):
    layer_model = model.module.model[0][layer-1]
    conv = layer_model.model[0]
    kernel_size = conv.kernel_size[0]
    padding = conv.padding[0]
    stride = conv.stride[0]
    
    n_patches = opt.random_crop_size // (opt.patch_size//2) - 1
    layer_inputs_raw = torch.load(os.path.join(opt.model_path, 'saved_input_layer_'+str(opt.save_vars_for_update_calc)))
    # padding as in forward path in VGG_like_Encoder.py
    pad = nn.ZeroPad2d(padding) # expects 4-dim input: b', c, y, x (b' = b*n_p_y*n_p_y)
    layer_inputs_pad = pad(layer_inputs_raw)
    s = layer_inputs_pad.shape
    layer_inputs = layer_inputs_pad.reshape(-1, n_patches, n_patches, s[1], s[2], s[3]) # b, n_p_y, n_p_x, c, y, x

    # get full, non-average-pooled output of the network
    _, out_full_, _, _ = layer_model.forward(layer_inputs_raw, None, 0, n_patches, n_patches, None) # b', c, n_p_y, n_p_x
    out_full = out_full_.reshape(-1, n_patches, n_patches, out_full_.shape[1], s[2]-2*padding, s[3]-2*padding) # b, n_p_y, n_p_x, c, y, x

    dW_ff = torch.zeros(conv.weight.shape) # c_post, c_pre, kernel, kernel
    for k in range(1, 6):
        which_update, context, z_p, z_n, rand_index, dloss_p, dloss_n = _load_activations(opt, layer, k)
        
        input_z_p = layer_inputs[:, (k + skip_step) :, :, :, :, :] # b, n_p_y (red.), n_p_x, c_pre, y, x (pre for pos. samples)
        input_z_n = input_z_p[rand_index, :, :, :, :, :].clone() # b, n_p_y (red.), n_p_x, c_pre, y, x (pre for neg. samples)
        input_context = layer_inputs[:, : -(k + skip_step), :, :, :, :] # b, n_p_y (red.), n_p_x, c_pre, y, x (pre for context)

        pre_z_p = input_z_p.unfold(4, kernel_size, stride).unfold(5, kernel_size, stride).permute(0, 1, 2, 4, 5, 3, 6, 7) # b, n_p_y (red.), n_p_x, y, x, c_pre, kernel_size, kernel_size
        pre_z_n = input_z_n.unfold(4, kernel_size, stride).unfold(5, kernel_size, stride).permute(0, 1, 2, 4, 5, 3, 6, 7)
        pre_context = input_context.unfold(4, kernel_size, stride).unfold(5, kernel_size, stride).permute(0, 1, 2, 4, 5, 3, 6, 7)
        
        # rho'(a), for ReLU -> sign function
        out_z_p = out_full[:, (k + skip_step) :, :, :, :, :] # b, n_p_y (red.), n_p_x, c_post, y, x
        out_z_n = out_z_p[rand_index, :, :, :, :, :].clone()
        out_c = out_full[:, : -(k + skip_step), :, :, :, :]

        post_z_p = torch.sign(out_z_p).permute(0, 1, 2, 4, 5, 3) # b, n_p_y (red.), n_p_x, y, x, c_post
        post_z_n = torch.sign(out_z_n).permute(0, 1, 2, 4, 5, 3)
        post_context = torch.sign(out_c).permute(0, 1, 2, 4, 5, 3)
          
        # post * pre
        dW_ff_k_p_pred = torch.einsum("bpqyxc, bpqyxdst -> bpqyxcdst", post_z_p.to('cuda'), pre_z_p.to('cuda')).mean(dim=(3,4)).to('cpu') # b, n_p_y, n_p_x, b, c_post, c_pre, k_s, k_s  (already av. over x and y positions)
        dW_ff_k_n_pred = torch.einsum("bpqyxc, bpqyxdst -> bpqyxcdst", post_z_n.to('cuda'), pre_z_n.to('cuda')).mean(dim=(3,4)).to('cpu') 
        dW_ff_k_retro = torch.einsum("bpqyxc, bpqyxdst -> bpqyxcdst", post_context.to('cuda'), pre_context.to('cuda')).mean(dim=(3,4)).to('cpu')

        # * "dendrite" (using transposed Wpred as Wretro!!)
        # In InfoNCE_Loss.py W_k is implemented with z as input -> W_k = W_retro!
        W_retro = copy.deepcopy(model.module.model[0][layer-1].loss.W_k[k-1].to('cpu')) # k - 1 because of zero-indexing!
        W_pred = copy.deepcopy(W_retro)
        W_pred.weight.data = W_retro.weight.permute(1, 0, 2, 3).clone().detach()

        pred = W_pred.forward(context.permute(2,3,0,1)).permute(0,2,3,1) # prediction (same for pos and neg): b, n_p_y (red.), n_p_x, c_post
        retro_p = W_retro.forward(z_p.permute(2,3,0,1)).permute(0,2,3,1) # retrodiction for pos. sample
        retro_n = W_retro.forward(z_n.permute(2,3,0,1)).permute(0,2,3,1) # retrodiction for neg. sample

        dW_ff_k_p_pred = pred.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dW_ff_k_p_pred
        dW_ff_k_p_retro = retro_p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dW_ff_k_retro
        dW_ff_k_n_pred = pred.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dW_ff_k_n_pred
        dW_ff_k_n_retro = retro_n.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dW_ff_k_retro

        # add contribution of pred and retro
        dW_ff_k_p = (dW_ff_k_p_pred + dW_ff_k_p_retro).permute(1,2,0,3,4,5,6) # n_p_y (red.), n_p_x, b, c_post, c_pre, kernel_size, kernel_size
        dW_ff_k_n = (dW_ff_k_n_pred + dW_ff_k_n_retro).permute(1,2,0,3,4,5,6)

        # --detach_c case
        # dW_ff_k_p = dW_ff_k_p_pred.permute(1,2,0,3,4,5,6) # n_p_y (red.), n_p_x, b, c_post, c_pre, kernel_size, kernel_size
        # dW_ff_k_n = dW_ff_k_n_pred.permute(1,2,0,3,4,5,6)
    
        # * "gamma"
        dW_ff_k_p = dloss_p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dW_ff_k_p # n_p_y (red.), n_p_x, b, c_post, c_pre, kernel_size, kernel_size
        dW_ff_k_n = dloss_n.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dW_ff_k_n

        dW_ff_k = _add_pos_and_neg_dW(which_update, dW_ff_k_p, dW_ff_k_n)
        dW_ff += dW_ff_k.clone()
        
    dW_ff /= opt.prediction_step
    
    return dW_ff

def compare_W_ff(opt, model):
    layer = opt.save_vars_for_update_calc

    dW_ff = _get_dW_ff(opt, layer)

    grad_W_ff = model.module.model[0][layer-1].model[0].weight.grad.squeeze().clone().detach().to('cpu')
    # model.module.model[0][layer-1].model[0].bias.grad

    diff_ff = dW_ff - grad_W_ff
    d_ff = diff_ff.norm() / (dW_ff.norm() + grad_W_ff.norm())

    return diff_ff, d_ff, grad_W_ff, dW_ff
    
    

##############################################################################################################
# Toy examples

def _get_loss(Wpred, z, c):
    Wpred_z = Wpred.forward(z).permute(2,3,0,1) # y, x, b, c
    scores = torch.matmul(c.permute(2,3,0,1).unsqueeze(-2), Wpred_z.unsqueeze(-1)).squeeze() # y, x, b
    ones = 0.1 * torch.ones(size=scores.shape, dtype=torch.float32)
    zeros = torch.zeros(size=scores.shape, dtype=torch.float32)
    loss = torch.where(scores < ones, ones - scores, zeros) # y, x, b

    return loss

def _compare_Wpred_toy(c, z, loss, Wpred):
    dWpred_ = torch.einsum("yxbc, yxbd -> yxbcd", c.permute(2,3,0,1), z.permute(2,3,0,1)) # y, x, b, c, c
    dloss = - torch.sign(loss) # y, x, b
    dWpred_ = dloss.unsqueeze(-1).unsqueeze(-1) * dWpred_
    dWpred = dWpred_.mean(dim=(0,1,2))
    
    # get grad
    Wpred_grad = Wpred.weight.grad.squeeze().clone()

    diff = dWpred - Wpred_grad
    d = diff.norm() / (dWpred.norm() + Wpred_grad.norm())

    return diff, d, Wpred_grad, dWpred

def _compare_W_ff_toy(c, z, c_full, z_full, loss, Wpred, layer, in_z_flat, in_c_flat, s):
    dloss = - torch.sign(loss) # y, x, b

    # zero pad + unfold
    pad = nn.ZeroPad2d(1) # expects 4-dim input: b', c, y, x
    in_z_pad = pad(in_z_flat).reshape(s[0], s[1], s[2], -1, s[4]+2, s[5]+2)  # b, n_p_y, n_p_x, c, y, x
    in_c_pad = pad(in_c_flat).reshape(s[0], s[1], s[2], -1, s[4]+2, s[5]+2)  # +2 because of padding

    pre_z = in_z_pad.unfold(4, 3, 1).unfold(5, 3, 1).permute(0, 1, 2, 4, 5, 3, 6, 7) # b, n_p_y, n_p_x, y, x, c_pre, kernel_size, kernel_size
    pre_c = in_c_pad.unfold(4, 3, 1).unfold(5, 3, 1).permute(0, 1, 2, 4, 5, 3, 6, 7)

    # sign post    # z_full: b, n_p_y, n_p_x, c_post, y, x
    post_z = torch.sign(z_full).permute(0, 1, 2, 4, 5, 3) # b, n_p_y, n_p_x, y, x, c_post
    post_c = torch.sign(c_full).permute(0, 1, 2, 4, 5, 3) 
    
    # pre * post
    dW_ff_pred_ = torch.einsum("bpqyxc, bpqyxdst -> bpqyxcdst", post_z, pre_z).permute(1,2,0,3,4,5,6,7,8) # n_p_y, n_p_x, b, y, x, c_post, c_pre, k_s, k_s
    dW_ff_retro_ = torch.einsum("bpqyxc, bpqyxdst -> bpqyxcdst", post_c, pre_c).permute(1,2,0,3,4,5,6,7,8)
    dW_ff_pred = dW_ff_pred_.mean(dim=(3,4)) # n_p_y, n_p_x, b, c_post, c_pre, k_s, k_s  (mean over x and y positions)
    dW_ff_retro = dW_ff_retro_.mean(dim=(3,4))

    # dendrite
    W_retro = copy.deepcopy(Wpred) # k - 1 because of zero-indexing!
    W_pred = copy.deepcopy(W_retro)
    W_pred.weight.data = W_retro.weight.permute(1, 0, 2, 3).clone().detach()

    pred = W_pred.forward(c).permute(2,3,0,1) # prediction: n_p_y, n_p_x, b, c_post
    retro = W_retro.forward(z).permute(2,3,0,1) # retrodiction
    
    dW_ff_pred = pred.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dW_ff_pred
    dW_ff_retro = retro.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dW_ff_retro

    # add pred and retro
    dW_ff = dW_ff_pred + dW_ff_retro # n_p_y, n_p_x, b, c, c, k, k

    # loss gating
    dW_ff = dloss.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dW_ff

    # mean over y, x, b
    dW_ff = dW_ff.mean(dim=(0,1,2)) # c, c, k, k 
    
    # get grad
    W_ff_grad = layer[0].weight.grad.squeeze().clone()

    diff_ff = dW_ff - W_ff_grad
    d_ff = diff_ff.norm() / (dW_ff.norm() + W_ff_grad.norm())

    return diff_ff, d_ff, W_ff_grad, dW_ff

def toyex():
    # define network and inputs
    W_ff = nn.Conv2d(3, 5, kernel_size=3, padding=1)
    nonlin = nn.ReLU()
    layer = nn.Sequential(*[W_ff, nonlin])
    Wpred = nn.Conv2d(5, 5, 1, bias=False)

    x_z = F.relu(torch.randn(10,3,8,12)) # b, c_in, y, x
    x_c = F.relu(torch.rand(x_z.shape))
    # extract patches
    in_z = x_z.unfold(2, 4, 2).unfold(3, 4, 2).permute(0, 2, 3, 1, 4, 5) # b, n_p_y, n_p_x, c_in, y, x
    in_c = x_c.unfold(2, 4, 2).unfold(3, 4, 2).permute(0, 2, 3, 1, 4, 5)
    s = in_z.shape

    in_z_flat = in_z.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5]) # b * n_p_y, n_p_x, c_in, y, x
    in_c_flat = in_c.reshape(s[0]*s[1]*s[2], s[3], s[4], s[5])

    # forward path
    out_z = layer.forward(in_z_flat) # b', c_out, y, x
    out_c = layer.forward(in_c_flat)

    z = F.adaptive_avg_pool2d(out_z, 1).squeeze().reshape(s[0], s[1], s[2], -1).permute(0,3,1,2) # b, c_out, n_p_y, n_p_x
    c = F.adaptive_avg_pool2d(out_c, 1).squeeze().reshape(s[0], s[1], s[2], -1).permute(0,3,1,2)

    z_full = out_z.reshape(s[0], s[1], s[2], -1, s[4], s[4]) # b, n_p_y, n_p_x, c, y, x
    c_full = out_c.reshape(s[0], s[1], s[2], -1, s[4], s[4]) # b, n_p_y, n_p_x, c, y, x

    loss = _get_loss(Wpred, z, c)
    l = loss.mean()
    
    # backward (calculate gradients)
    l.backward()

    ## manual update Wpred
    diff, d, Wpred_grad, dWpred = _compare_Wpred_toy(c, z, loss, Wpred)
    print("d for Wpred: ", d)
    
    ## manual update W_ff
    diff_ff, d_ff, W_ff_grad, dW_ff = _compare_W_ff_toy(c, z, c_full, z_full, loss, Wpred, layer, in_z_flat, in_c_flat, s)
    print("d for W_ff: ", d_ff)

    embed()
   

##############################################################################################################

if __name__ == "__main__":
    
    #toyex()

    opt = arg_parser.parse_args()
    
    if opt.model_splits != 6:
        raise Exception("Only works for layer-wise CLAPP, i.e. model_splits = 6 for 6 layers!")

    # check for layers 0 (1) and 2 (3) since the others have maxpooling layers!
    if opt.save_vars_for_update_calc != 1 and opt.save_vars_for_update_calc != 3:
        raise Exception("Comparison between updates and gradients only implemented for layers without trailing MaxPool layers (i.e. layers 1 & 3)")
    
    opt.training_dataset = "unlabeled"

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model, optimizer = load_vision_model.load_model_and_optimizer(opt)

    if opt.batch_size != opt.batch_size_multiGPU:
        raise Exception("Manual update comparison only supported for 1 GPU. Please use only 1 GPU")
    
    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt
    )

    # perform one training iter and save reps etc.
    train_iter(opt, model, train_loader)

    # check equivalence between grads and updates for Wpred
    print("checking equivalence between grads and updates for Wpred...")
    for k in range(1,6):
        diff, d, grad_Wpred, dWpred = compare_Wpred(opt, model, k)
        print("rel. difference between grad and update for Wpred_k for k=", k, ": ", d)
    
    # check equivalence between grads and updates for W_ff
    print("checking equivalence between grads and updates for W_ff...")
    diff_ff, d_ff, grad_W_ff, dW_ff = compare_W_ff(opt, model)
    print("rel. difference between grad and update for W_ff: ", d_ff)
    
    embed()
    
    