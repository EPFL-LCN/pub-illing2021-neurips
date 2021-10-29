# Partly taken from (30 July 2020)
# https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg11
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os

from CLAPPVision.vision.models import ContrastiveLoss, Supervised_Loss
from CLAPPVision.utils import model_utils

class VGG_like_Encoder(nn.Module):
    def __init__(
        self,
        opt,
        block_idx,
        blocks,
        in_channels,
        patch_size=16,
        overlap_factor=2,
        calc_loss=False,
    ):
        super(VGG_like_Encoder, self).__init__()
        self.encoder_num = block_idx
        self.opt = opt

        self.save_vars = self.opt.save_vars_for_update_calc == block_idx+1

        # Layer
        self.model = self.make_layers(blocks[block_idx], in_channels)

        # Params
        self.calc_loss = calc_loss

        self.overlap = overlap_factor
        self.increasing_patch_size = self.opt.increasing_patch_size
        if self.increasing_patch_size: # This is experimental... take care, this must be synced with architecture, i.e. number and position of downsampling layers (stride 2, e.g. pooling)
            if self.overlap != 2:
                raise ValueError("if --increasing_patch_size is true, overlap(_factor) has to be equal 2")
            patch_sizes = [4, 4, 8, 8, 16, 16]
            self.patch_size_eff = patch_sizes[block_idx]
            self.max_patch_size = max(patch_sizes)
            high_level_patch_sizes = [4, 4, 4, 4, 4, 2]
            self.patch_size = high_level_patch_sizes[block_idx]
        else:
            self.patch_size = patch_size
        
        reduced_patch_pool_sizes = [4, 4, 3, 3, 2, 1]
        if opt.reduced_patch_pooling:
            self.patch_average_pool_out_dim = reduced_patch_pool_sizes[block_idx]
        else:
            self.patch_average_pool_out_dim = 1

        self.predict_module_num = self.opt.predict_module_num
        self.extra_conv = self.opt.extra_conv
        self.inpatch_prediction = self.opt.inpatch_prediction
        self.inpatch_prediction_limit = self.opt.inpatch_prediction_limit
        self.asymmetric_W_pred = self.opt.asymmetric_W_pred

        if opt.gradual_prediction_steps:
            prediction_steps = min(block_idx+1, self.opt.prediction_step)
        else:
            prediction_steps = self.opt.prediction_step

        def get_last_index(block):
            if block[-1] == 'M':
                last_ind = -2
            else:
                last_ind = -1
            return last_ind

        last_ind = get_last_index(blocks[block_idx])
        self.in_planes = blocks[block_idx][last_ind]
        # in_channels_loss: z, out_channels: c
        if self.predict_module_num=='-1' or self.predict_module_num=='both':
            if self.encoder_num == 0: # exclude first module
                in_channels_loss = self.in_planes
                if opt.reduced_patch_pooling:
                    in_channels_loss *= reduced_patch_pool_sizes[block_idx] ** 2
            else:
                last_ind_block_below = get_last_index(blocks[block_idx-1])
                in_channels_loss = blocks[block_idx-1][last_ind_block_below]
                if opt.reduced_patch_pooling:
                    in_channels_loss *= reduced_patch_pool_sizes[block_idx-1] ** 2
        else:
            in_channels_loss = self.in_planes
            if opt.reduced_patch_pooling:
                in_channels_loss *= reduced_patch_pool_sizes[block_idx] ** 2
        
        # Optional extra conv layer to increase rec. field size
        if self.extra_conv and self.encoder_num < 3:
            self.extra_conv_layer = nn.Conv2d(self.in_planes, self.in_planes, stride=3, kernel_size=3, padding=1)

        # in_channels_loss: z, out_channels: c
        if self.predict_module_num == '-1b':
            if self.encoder_num == len(blocks)-1: # exclude last module
                out_channels = self.in_planes
                if opt.reduced_patch_pooling:
                    out_channels *= reduced_patch_pool_sizes[block_idx] ** 2
            else:
                last_ind_block_above = get_last_index(blocks[block_idx+1])
                out_channels = blocks[block_idx+1][last_ind_block_above]
                if opt.reduced_patch_pooling:
                    out_channels *= reduced_patch_pool_sizes[block_idx+1] ** 2
        else:
            out_channels = self.in_planes
            if opt.reduced_patch_pooling:
                out_channels *= reduced_patch_pool_sizes[block_idx] ** 2


        # Loss module; is always present, but only gets used when training CLAPPVision modules
        # in_channels_loss: z, out_channels: c
        if self.opt.loss == 0:
            self.loss = ContrastiveLoss.ContrastiveLoss(
                opt,
                in_channels=in_channels_loss, # z
                out_channels=out_channels, # c
                prediction_steps=prediction_steps,
                save_vars=self.save_vars
            )
            if self.predict_module_num == 'both':
                self.loss_same_module = ContrastiveLoss.ContrastiveLoss(
                    opt,
                    in_channels=in_channels_loss,
                    out_channels=in_channels_loss, # on purpose, cause in_channels_loss is layer below
                    prediction_steps=prediction_steps
                )
            if self.asymmetric_W_pred:
                self.loss_mirror = ContrastiveLoss.ContrastiveLoss(
                opt,
                in_channels=in_channels_loss,
                out_channels=out_channels,
                prediction_steps=prediction_steps
                )
        elif self.opt.loss == 1:
            self.loss = Supervised_Loss.Supervised_Loss(opt, in_channels_loss, True)
        else:
            raise Exception("Invalid option")
        

        # Optional recurrent weights, Experimental!
        if self.opt.inference_recurrence == 1 or self.opt.inference_recurrence == 3: # 1 - lateral recurrence within layer
            self.recurrent_weights = nn.Conv2d(self.in_planes, self.in_planes, 1, bias=False)
        if self.opt.inference_recurrence == 2 or self.opt.inference_recurrence == 3: # 2 - feedback recurrence, 3 - both, lateral and feedback recurrence
            if self.encoder_num < len(blocks)-1: # exclude last module
                last_ind_block_above = get_last_index(blocks[block_idx+1])
                rec_dim_block_above = blocks[block_idx+1][last_ind_block_above]
                self.recurrent_weights_fb = nn.Conv2d(rec_dim_block_above, self.in_planes, 1, bias=False)

        if self.opt.weight_init:
            raise NotImplementedError("Weight init not implemented for vgg")
    
    def make_layers(self, block, in_channels, batch_norm=False, inplace=False):
        layers = []
        for v in block:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=inplace)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=inplace)]
                in_channels = v
        return nn.Sequential(*layers)


    def forward(self, x, reps, t, n_patches_y, n_patches_x, label):
        # x: either input dims b, c, Y, X or (if coming from lower module which did unfolding, as variable z):  b * n_patches_y * n_patches_x, c, y, x

        # Input preparation, i.e unfolding into patches. Usually only needed for first module. More complicated for experimental increasing_patch_size option.
        if self.encoder_num in [0,2,4]: # [0,2,4,5] 
            # if increasing_patch_size is enabled, this has to be in sync with architecture and intended patch_size for respective module:
            # for every layer that increases the patch_size, the extra downsampling + unfolding has to be done!
            if self.encoder_num > 0 and self.increasing_patch_size:
                # undo unfolding of the previous module
                s1 = x.shape
                x = x.reshape(-1, n_patches_y, n_patches_x, s1[1], s1[2], s1[3]) # b, n_patches_y, n_patches_x, c, y, x
                # downsampling to get rid of the overlaps between paches of the previous module
                x = x[:,::2,::2,:,:,:] # b, n_patches_x_red, n_patches_y_red, c, y, x. 
                s = x.shape
                x = x.permute(0,3,2,5,1,4).reshape(s[0],s[3],s[2],s[5],s[1]*s[4]).permute(0,1,4,2,3).reshape(s[0],s[3],s[1]*s[4],s[2]*s[5]) # b, c, Y, X

            if self.encoder_num == 0 or self.increasing_patch_size:                
                x = ( # b, c, y, x
                    x.unfold(2, self.patch_size, self.patch_size // self.overlap) # b, c, n_patches_y, x, patch_size
                    .unfold(3, self.patch_size, self.patch_size // self.overlap) # b, c, n_patches_y, n_patches_x, patch_size, patch_size
                    .permute(0, 2, 3, 1, 4, 5) # b, n_patches_y, n_patches_x, c, patch_size, patch_size
                )
                n_patches_y = x.shape[1]
                n_patches_x = x.shape[2]
                x = x.reshape(
                    x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
                ) # b * n_patches_y * n_patches_x, c, patch_size, patch_size

        # Main encoding step
        # forward through self.model is split into (conv)/(nonlin + pool) due to (optional) recurrence
        # assuming arch = [128, 256, 'M', 256, 512, 'M', 1024, 'M', 1024, 'M']
        if self.opt.inference_recurrence > 0: # in case of recurrence
            if self.opt.model_splits == 6:
                split_ind = 1
            elif self.opt.model_splits == 3 or self.opt.model_splits == 1:
                split_ind = -2
            else:
                raise NotImplementedError("Recurrence is only implemented for model_splits = 1, 3 or 6")
        else: # without recurrence split does not really matter, arbitrily choose 1
            split_ind = 1

        if self.save_vars: # save input for (optional) manual update calculation
            torch.save(x, os.path.join(self.opt.model_path, 'saved_input_layer_'+str(self.opt.save_vars_for_update_calc)))

        # 1. Apply encoding weights of model (e.g. conv2d layer)
        z = self.model[:split_ind](x) # b * n_patches_y * n_patches_x, c, y, x

        # 2. Add (optional) recurrence if present
        # expand dimensionality if rec comes from layer after one or several (2x2 strided, i.e. downsampled) pooling layer(s). tensor.repeat_interleave() would do, but not available in pytorch 1.0.0
        def expand_2_by_2(rec):
            srec = rec.shape
            return (
                rec.unfold(2,1,1).repeat((1,1,1,1,2)).permute(0,1,2,4,3).reshape(srec[0],srec[1],2*srec[2],srec[3])
                    .unfold(3,1,1).repeat((1,1,1,1,2)).reshape(srec[0],srec[1],2*srec[2],2*srec[3])
            )
        if t > 0: # only apply rec if iteration is not the first one (is not entered if recurrence is off since then t = 0)
            if self.opt.inference_recurrence == 1 or self.opt.inference_recurrence == 3: # 1 - lateral recurrence within layer
                rec = self.recurrent_weights(reps[self.encoder_num].clone().detach()) # Detach input to implement e-prop like BPTT
                while z.shape != rec.shape: # if rec comes from strided pooling layer
                    rec = expand_2_by_2(rec)
                z += rec
            if self.opt.inference_recurrence == 2 or self.opt.inference_recurrence == 3: # 2 - feedback recurrence, 3 - both, lateral and feedback recurrence
                if self.encoder_num < len(reps)-1: # exclude last module
                    rec_fb = self.recurrent_weights_fb(reps[self.encoder_num+1].clone().detach()) # Detach input to implement e-prop like BPTT
                    while z.shape != rec_fb.shape: # if rec comes from strided pooling layer
                        rec_fb = expand_2_by_2(rec_fb)
                    z += rec_fb
        
        # 3. Apply nonlin and 'rest' of model (e.g. ReLU, MaxPool etc...)
        z = self.model[split_ind:](z) # b * n_patches_y * n_patches_x, c, y, x

        # Optional extra conv layer with downsampling (stride > 1) here to increase receptive field size ###
        if self.extra_conv and self.encoder_num < 3:
            dec = self.extra_conv_layer(z)
            dec = F.relu(dec, inplace=False)
        else:
            dec = z

        # Optional in-patch prediction
        # if opt: change CPC task to smaller scale prediction (within patch -> smaller receptive field)
        # by extra unfolding + "cropping" (to avoid overweighing lower layers and memory overflow)
        if self.inpatch_prediction and self.encoder_num < self.inpatch_prediction_limit:
            extra_patch_size = [2 for _ in range(self.inpatch_prediction_limit)]
            extra_patch_steps = [1 for _ in range(self.inpatch_prediction_limit)]
            
            dec = dec.reshape(-1, n_patches_x, n_patches_y, dec.shape[1], dec.shape[2], dec.shape[3]) # b, n_patches_y, n_patches_x, c, y, x
            # random "cropping"/selecting of patches that will be extra unfolded
            extra_crop_size = [n_patches_x // 2 for _ in range(self.inpatch_prediction_limit)]
            inds = np.random.randint(0, n_patches_x - extra_crop_size[self.encoder_num], 2)
            dec = dec[:, inds[0]:inds[0]+extra_crop_size[self.encoder_num], inds[1]:inds[1]+extra_crop_size[self.encoder_num],:,:,:]
            
            # extra unfolding            
            dec = (
                dec.unfold(4, extra_patch_size[self.encoder_num], extra_patch_steps[self.encoder_num])
                .unfold(5, extra_patch_size[self.encoder_num], extra_patch_steps[self.encoder_num]) # b, n_patches_y, n_patches_x, c, n_extra_patches, n_extra_patches, extra_patch_size, extra_patch_size
                .permute(0, 1, 2, 4, 5, 3, 6, 7) # b, n_patches_y(_reduced), n_patches_x(_reduced), n_extra_patches, n_extra_patches, c, extra_patch_size, extra_patch_size
            )
            n_extra_patches = dec.shape[3]
            dec = dec.reshape(dec.shape[0] * dec.shape[1] * dec.shape[2] * dec.shape[3] * dec.shape[4], dec.shape[5], dec.shape[6], dec.shape[7])
            # b * n_patches_y(_reduced) * n_patches_x(_reduced) * n_extra_patches * n_extra_patches, c, extra_patch_size, extra_patch_size

        # Pool over patch
        # in original CPC/GIM, pooling is done over whole patch, i.e. output shape 1 by 1
        out = F.adaptive_avg_pool2d(dec, self.patch_average_pool_out_dim) # b * n_patches_y(_reduced) * n_patches_x(_reduced) (* n_extra_patches * n_extra_patches), c, x_pooled, y_pooled
        # Flatten over channel and pooled patch dimensions x_pooled, y_pooled:
        out = out.reshape(out.shape[0], -1) # b * n_patches_y(_reduced) * n_patches_x(_reduced) (* n_extra_patches * n_extra_patches),  c * y_pooled * x_pooled
        
        if self.inpatch_prediction and self.encoder_num < self.inpatch_prediction_limit:
            n_p_x, n_p_y = n_extra_patches, n_extra_patches
        else:
            n_p_x, n_p_y = n_patches_x, n_patches_y

        out = out.reshape(-1, n_p_y, n_p_x, out.shape[1]) # b, n_patches_y, n_patches_x, c * y_pooled * x_pooled OR  b * n_patches_y(_reduced) * n_patches_x(_reduced), n_extra_patches, n_extra_patches, c * y_pooled * x_pooled
        out = out.permute(0, 3, 1, 2).contiguous() # b, c * y_pooled * x_pooled, n_patches_y, n_patches_x  OR  b * n_patches_y(_reduced) * n_patches_x(_reduced), c * y_pooled * x_pooled, n_extra_patches, n_extra_patches
  
        return out, z, n_patches_y, n_patches_x

    # crop feature map such that the loss always predicts/averages over same amount of patches (as the last one)
    def random_spatial_crop(self, out, n_patches_x, n_patches_y):
        n_patches_x_crop = n_patches_x // (self.max_patch_size // self.patch_size_eff)
        n_patches_y_crop = n_patches_y // (self.max_patch_size // self.patch_size_eff)
        if n_patches_x == n_patches_x_crop:
            posx = 0
        else:
            posx = np.random.randint(0, n_patches_x - n_patches_x_crop + 1)
        if n_patches_y == n_patches_y_crop:
            posy = 0
        else:
            posy = np.random.randint(0, n_patches_y - n_patches_y_crop + 1)
        out = out[:, :, posy:posy+n_patches_y_crop, posx:posx+n_patches_x_crop]
        return out

    def evaluate_loss(self, outs, cur_idx, label, gating=None):
        accuracy = torch.zeros(1)
        gating_out = None
        if self.calc_loss and self.opt.loss == 0:
            # Special cases of predicting module below or same module and below ('both')
            if self.predict_module_num=='-1' or self.predict_module_num=='both': # gating not implemented here!
                if self.asymmetric_W_pred:
                    raise NotImplementedError("asymmetric W not implemented yet for predicting lower layers!")
                if self.encoder_num==0: # normal loss for first module
                    loss, loss_gated, gating_out = self.loss(outs[cur_idx], outs[cur_idx], gating=gating) # z, c
                else:
                    loss, loss_gated, _ = self.loss(outs[cur_idx-1], outs[cur_idx]) # z, c
                    if self.predict_module_num=='both':
                        loss_intralayer, _, _ = self.loss_same_module(outs[cur_idx], outs[cur_idx])
                        loss = 0.5 * (loss + loss_intralayer)
            
            elif self.predict_module_num=='-1b':
                if self.asymmetric_W_pred:
                    raise NotImplementedError("asymmetric W not implemented yet for predicting lower layers!")
                if self.encoder_num == len(outs)-1: # normal loss for last module
                    loss, loss_gated, gating_out = self.loss(outs[cur_idx], outs[cur_idx], gating=gating) # z, c
                else:
                    loss, loss_gated, _ = self.loss(outs[cur_idx], outs[cur_idx+1]) # z, c
            # Normal case for prediction within same layer
            else: 
                if self.asymmetric_W_pred: # u = z*W_pred*c -> u = drop_grad(z)*W_pred1*c + z*W_pred2*drop_grad(c)
                    if self.opt.contrast_mode != 'hinge':
                        raise ValueError("asymmetric_W_pred only implemented for hinge contrasting!")
                    
                    loss, loss_gated, _ = self.loss(outs[cur_idx], outs[cur_idx].clone().detach(), gating=gating) # z, detach(c)
                    
                    
                    loss_mirror, loss_mirror_gated, _ = self.loss_mirror(outs[cur_idx].clone().detach(), outs[cur_idx], gating=gating) # detach(z), c
                    
                    loss = loss + loss_mirror
                    loss_gated = loss_gated + loss_mirror_gated
                else:
                    loss, loss_gated, gating_out = self.loss(outs[cur_idx], outs[cur_idx], gating=gating) # z, c
    
        elif self.calc_loss and self.opt.loss == 1: # supervised loss
            loss, accuracy = self.loss(outs[cur_idx], label)
            loss_gated, gating_out = -1, -1
        else: # only forward pass for downstream classification
            loss, loss_gated, accuracy, gating_out = None, None, None, None

        return loss, loss_gated, accuracy, gating_out
