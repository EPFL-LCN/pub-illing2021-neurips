import torch
import torch.nn as nn

from CLAPPVision.vision.models import VGG_like_Encoder

class FullVisionModel(torch.nn.Module):
    def __init__(self, opt, calc_loss):
        super().__init__()
        self.opt = opt
        self.contrastive_samples = self.opt.negative_samples
        if self.opt.current_rep_as_negative:
            print("Contrasting against current representation (i.e. only one negative sample)")
        else:
            print("Contrasting against ", self.contrastive_samples, " negative sample(s)")
        self.calc_loss = calc_loss
        self.encoder_type = self.opt.encoder_type
        print("Using ", self.encoder_type, " encoder")
        self.predict_module_num = self.opt.predict_module_num
        self.increasing_patch_size = self.opt.increasing_patch_size
        if self.predict_module_num=='-1':
            print("Predicting lower module, 1st module predicting same module")
        elif self.predict_module_num=='both':
            print("Predicting both, same and lower module")
        elif self.predict_module_num=='-1b':
            print("Predicting lower module, last module predicting same module")

        if self.opt.inference_recurrence == 0: # 0 - no recurrence
            self.recurrence_iters = 0
        else: # 1 - lateral recurrence within layer, 2 - feedback recurrence, 3 - both, lateral and feedback recurrence
            self.recurrence_iters = self.opt.recurrence_iters

        self.model, self.encoder = self._create_full_model(opt)

        print(self.model)

    def _create_full_model(self, opt):
        if self.encoder_type=='vgg_like':
            full_model, encoder = self._create_full_model_vgg(opt)
        else:
            raise Exception("Invalid encoder option")

        return full_model, encoder


    def _create_full_model_vgg(self, opt):
        if type(opt.patch_size) == int:
            patch_sizes = [opt.patch_size for _ in range(opt.model_splits)]
        else:
            patch_sizes = opt.patch_size

        arch = [128, 256, 'M', 256, 512, 'M', 1024, 'M', 1024, 'M']
        if opt.model_splits == 1:
            blocks = [arch]
        elif opt.model_splits == 2:
            blocks = [arch[:4], arch[4:]]
        elif opt.model_splits == 4:
            blocks = [arch[:4], arch[4:6], arch[6:8], arch[8:]]
        elif opt.model_splits == 3: 
            blocks = [arch[:3], arch[3:6], arch[6:]]
        elif opt.model_splits == 6:
            blocks = [arch[:1], arch[1:3], arch[3:4], arch[4:6], arch[6:8], arch[8:]]
        else:
            raise NotImplementedError

        full_model = nn.ModuleList([])
        encoder = nn.ModuleList([])

        if opt.grayscale:
            input_dims = 1
        else:
            input_dims = 3

        output_dims = arch[-2] * 4

        for idx, _ in enumerate(blocks):
            if idx==0:
                in_channels = input_dims
            else:
                if blocks[idx-1][-1] == 'M':
                    in_channels = blocks[idx-1][-2]
                else:
                    in_channels = blocks[idx-1][-1]

            encoder.append(
                VGG_like_Encoder.VGG_like_Encoder(opt,
                idx,
                blocks,
                in_channels,
                calc_loss=False,
                patch_size=patch_sizes[idx],
                )
            )

        full_model.append(encoder)

        return full_model, encoder

    ###########################################################################################################
    # forward

    def forward(self, x, label, n=3):
        # n: until which module to perform the forward pass
        model_input = x

        if self.opt.device.type != "cpu":
            cur_device = x.get_device()
        else:
            cur_device = self.opt.device

        n_patches_x, n_patches_y = None, None

        
        outs = []

        if n==-1: # return (reshaped/flattened) input image, for direct classification
            s = model_input.shape # b, in_channels, y, x
            h = model_input.reshape(s[0], s[1]*s[2]*s[3]).unsqueeze(-1).unsqueeze(-1) # b, in_channels*y*x
        else:
            reps = None
            for t in range(self.recurrence_iters+1): # 0-th iter for normal feedforward pass
                model_input = x
                acts = []
                # forward loop through modules
                for idx, module in enumerate(self.encoder[:n]):
                    # block gradient of h at some point -> should be blocked after one module since input was detached
                    h, z, n_patches_y, n_patches_x = module(
                        model_input, reps, t, n_patches_y, n_patches_x, label
                    )
                    # detach z to make sure no gradients are flowing in between modules
                    # we can detach z here, as for the CPC model the loop is only called once and h is forward-propagated
                    model_input = z.clone().detach() # full module output 
                    acts.append(model_input) # needed for optional recurrence
                    if t == self.recurrence_iters:
                        outs.append(h) # out: mean pooled per patch
                    
                reps = acts

        loss, loss_gated, accuracies = self.evaluate_losses(outs, label, cur_device, n=n)
        
        c = None # Can be used if context is of different kind than h (e.g. output of recurrent layer)
    
        return loss, loss_gated, c, h, accuracies

    def evaluate_losses(self, outs, label, cur_device, n = 3): 
        loss = torch.zeros(1, self.opt.model_splits, device=cur_device) # first dimension for multi-GPU training
        loss_gated = torch.zeros(1, self.opt.model_splits, device=cur_device) # first dimension for multi-GPU training
        accuracies = torch.zeros(1, self.opt.model_splits, device=cur_device) # first dimension for multi-GPU training

        # loop BACKWARDS through module outs and calculate losses
        # backward loop is necessary because of potential feedback gating!
        for idx in range(n-1, -1, -1): # backward loop: upper, lower, step
            if self.opt.feedback_gating:
                if idx == self.opt.model_splits-1: # no gating for highest layer
                    gating = None
            else:
                gating = None
            
            cur_loss, cur_loss_gated, cur_accuracy, gating = self.encoder[idx].evaluate_loss(outs, idx, label, gating=gating)

            if cur_loss is not None:
                loss[:, idx] = cur_loss
                loss_gated[:, idx] = cur_loss_gated
                accuracies[:, idx] = cur_accuracy

        return loss, loss_gated, accuracies

    def switch_calc_loss(self, calc_loss):
        # by default models are set to not calculate the loss as it is costly
        # this function can enable the calculation of the loss for training
        self.calc_loss = calc_loss
        if self.opt.model_splits == 1 and self.opt.loss == 0:
            self.encoder[-1].calc_loss = calc_loss

        if self.opt.model_splits == 1 and self.opt.loss == 1:
            self.encoder[-1].calc_loss = calc_loss

        if self.opt.model_splits > 1:
            if self.opt.train_module != self.opt.model_splits:
                cont = input("WARNING: model_splits > 1 and train_module != model_splits."
                " (this could mean that not all modules are trained; ignore when training classifier)."
                " Please think again if you really want that and enter 'y' to continue: ")
                if cont == "y":
                    return
                else:
                    raise ValueError("Interrupting...")

            if self.opt.train_module == self.opt.model_splits:
                for i, layer in enumerate(self.encoder):
                    layer.calc_loss = calc_loss
            else:
                self.encoder[self.opt.train_module].calc_loss = calc_loss
