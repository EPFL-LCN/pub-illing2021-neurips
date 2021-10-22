
# Call as downstream classification, e.g.
# python -m GreedyInfoMax.vision.visualise --model_path ./logs/your_simulation --model_num 299 --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --module_num 6 --batch_size 100

################################################################################

import torch
import numpy as np
from numpy.random import choice
import time
import os
import code
import sklearn
from IPython import embed
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

## own modules
from GreedyInfoMax.vision.data import get_dataloader
from GreedyInfoMax.vision.arg_parser import arg_parser
from GreedyInfoMax.vision.models import load_vision_model
from GreedyInfoMax.utils import logger, utils


def load_model_and_data(opt):
    add_path_var = "linear_model"

    arg_parser.create_log_path(opt, add_path_var=add_path_var)
    opt.training_dataset = "train"

    # load pretrained model
    # cannot switch opt.reduced_patch_pooling = False here because otherwise W_preds sizes don't match
    context_model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False
    )
    context_model.module.switch_calc_loss(False)
    
    ## model_type=2 is supervised model which trains entire architecture; otherwise just extract features
    if opt.model_type != 2:
        context_model.eval()
        
    if opt.module_num==-1:
        print("CAREFUL! Training classifier directly on input image! Model is ignored and returns the (flattened) input images!")

    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt)

    return context_model, train_loader, test_loader

def get_representations(opt, model, data_loader, module=None, reload=True):
    if type(data_loader.dataset.transform.transforms[0].transforms[0]) != transforms.transforms.CenterCrop:
        raise Exception("Data loader should use deterministic cropping (Center Crop) for image patch visualisation!")
    
    if module == None:
        module = opt.module_num

    if reload:
        print("Reload representations...")
        (inputs, reps, targets) = torch.load(os.path.join(opt.model_path, 'saved_reps_module'+str(module)), map_location=torch.device('cpu'))
    else:
        print("Calculate representations...")
        inputs = []
        reps = []
        targets = []
        for step, (img, target) in enumerate(data_loader):
            print("batch number: ", step, " of ", len(data_loader))
            model_input = img.to(opt.device)
            if opt.model_type == 2:  ## fully supervised training
                    _, _, z = model(model_input)
            else:
                with torch.no_grad():
                    _, _, _, z, _ = model(model_input, target, n=module)
                
            inputs.append(model_input)
            reps.append(z.detach())
            targets.append(target)

        inputs = torch.cat(inputs).cpu()
        reps = torch.cat(reps).cpu()
        targets = torch.cat(targets).cpu()

        torch.save((inputs, reps, targets), os.path.join(opt.model_path, 'saved_reps_module'+str(module)))

    return inputs, reps, targets

##############################################################################################################
# Visualisation of learned "Manifold" by t-SNE embedding

def tSNE(opt, inputs, reps, targets, class_names, n_points = None):
    print("Doing t-SNE...")

    n_samples = targets.shape[0]
    if n_points == None:
        n_points = n_samples

    d_inputs = inputs.reshape(n_samples, -1)

    reps_m = torch.mean(reps,(2,3)) # spatial mean pooling
    d_reps = reps_m.reshape(n_samples, -1) 

    tsne_inputs = sklearn.manifold.TSNE(perplexity = 50)
    tsne_reps = sklearn.manifold.TSNE(perplexity = 50)
    
    #t_inputs = tsne_inputs.fit_transform(d_inputs[:n_points,:])
    t_reps = tsne_reps.fit_transform(d_reps[:n_points,:])

    #tSNE_plot(opt, t_inputs, targets[:n_points], class_names, fig_name_ext = 'input')
    tSNE_plot(opt, t_reps, targets[:n_points], class_names)

def tSNE_plot(opt, t_data, targets, class_names, fig_name_ext = '', markersize = 2, plot_legend = False):
    plt.figure()
    for class_index in range(10): # loop over classes
        inds = targets == class_index
        if sum(inds) > 0: # exclude empty sets
            t_data_plot = t_data[inds, :]
            plt.scatter(t_data_plot[:,0],t_data_plot[:,1], label = class_names[class_index], s=markersize)
    plt.axis('off')
    if plot_legend:
        plt.legend(markerscale=3)
    plt.savefig(os.path.join(opt.model_path, 'tSNE_module'+str(opt.module_num)+fig_name_ext+'.pdf'))

##############################################################################################################
# Visualisation of learned "Manifold" by looking at neighbour encodings in feature space 

def unravel_index(indices: torch.LongTensor, shape) -> torch.LongTensor:
    """Converts flat indices into unraveled coordinates in a target shape.
    This is a `torch` implementation of `numpy.unravel_index`.
    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord

def patch_neighbouranalysis(opt, imgs, reps, 
        n_examples = 5, n_neighbours = 10, patch_size = 16, patch_spacing = 8, center_crop_margin = 16, do_savefig = False):
    
    n_patches = reps.shape[-1]
    patch_coords = choice([i for i in range(n_patches)], size = (n_examples, 2), replace=True)
    
    ts, neighbours_list = find_neighbours(reps, n_examples, n_neighbours, patch_coords)
    
    plot_patches(opt, imgs, patch_coords, ts, neighbours_list, n_examples, n_neighbours, 
        patch_size = patch_size, patch_spacing = patch_spacing, center_crop_margin = center_crop_margin,  do_savefig =  do_savefig)

    
def find_neighbours(reps, n_examples, n_neighbours, patch_coords):
    # reps: b, c, x, y
    s = reps.shape

    ts = []
    neighbours_list = []
    for i in range(n_examples):
        t = np.random.randint(s[0])
        ts.append(t)
        rep_t = reps[t, :, patch_coords[i,0], patch_coords[i,1]].squeeze() # c (reference patch)

        rep = reps.permute(1, 0, 2, 3).reshape(s[1], -1).permute(1,0) # b*x*y, c (flattened patch list)
        
        l2_dif = torch.norm(rep - rep_t, dim=1) # b*x*y
        neighbours_flattened = l2_dif.argsort()
        neighbours_array = unravel_index(neighbours_flattened, (s[0], s[2], s[3])) # b*x*y, 3
        neighbours = neighbours_array[1:n_neighbours+1] # n_neighbours, 3 (time, x, y) (closest neighbours, first excluded since same patch)
        
        neighbours_list.append(neighbours)
    
    return ts, neighbours_list


def plot_patches(opt, imgs, patch_coords, ts, neighbours_list, n_examples, n_neighbours, 
        patch_size = 16, patch_spacing = 8, center_crop_margin = 8, plot_reference_patch = True, crop = False, do_savefig = False):
    # neighbours_list is list of 2dim arrays, each of which with dimensions: n_neighbours, 3 (time, x, y)
        
    def _add_patch_frame(img, p_coord, p_size, p_space, margin, color = 'black', extra_line_width = 2):
        def _set_pixel_values(img, c, value, p_coord, p_size, p_space, margin, extra_line_width):
            x0 = margin+p_coord[0]*p_space
            y0 = margin+p_coord[1]*p_space
            img[x0:x0+p_size+extra_line_width, y0:y0+extra_line_width, c] = value
            img[x0:x0+p_size+extra_line_width, y0+p_size:y0+p_size+extra_line_width, c] = value
            img[x0:x0+extra_line_width, y0:y0+p_size+extra_line_width, c] = value
            img[x0+p_size:x0+p_size+extra_line_width, y0:y0+p_size+extra_line_width, c] = value
            return img
        
        img = _set_pixel_values(img, [0,1,2], 0, p_coord, p_size, p_space, margin, extra_line_width) # black frame  
        if color == 'red':
            _set_pixel_values(img, [0], 255, p_coord, p_size, p_space, margin, extra_line_width) # red frame

        return img
    
    n_plots = n_neighbours
    if plot_reference_patch: # first one is reference patch itself
        n_plots += 1
    
    imgs_list = []
    for ex in range(n_examples):
        imgs_select = []

        if plot_reference_patch:
            img = imgs[ts[ex], :, :, :].copy().transpose(1,2,0) # full, uncropped color image x, y, c
            img = _add_patch_frame(img, patch_coords[ex, :], patch_size, patch_spacing, center_crop_margin, color = 'black')
            imgs_select.append(img)        
        for n in range(n_neighbours):
            img = imgs[neighbours_list[ex][n][0], :, :, :].copy().transpose(1,2,0) # full, uncropped color image x, y, c
            img = _add_patch_frame(img, neighbours_list[ex][n][1:], patch_size, patch_spacing, center_crop_margin, color = 'red')
            imgs_select.append(img)

        imgs_list.append(imgs_select)
    
    fig, axes = plt.subplots(nrows=n_examples, ncols=n_plots, sharex=True, sharey=True) # , gridspec_kw={'wspace': 0.05})
    fig.set_size_inches(10, 10)
    for i in range(n_examples):
        for j in range(n_plots): 
            ax = axes[i][j]
            if crop:
                ax.imshow(imgs_list[i][j][center_crop_margin:-center_crop_margin, center_crop_margin:-center_crop_margin])
            else:
                ax.imshow(imgs_list[i][j])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

    if do_savefig:
        plt.savefig(os.path.join(opt.model_path, 'patch_visualisation_module'+str(opt.module_num)+'.pdf'))        


##############################################################################################################
# Visualisation of neuron receptive fields by plotting maximally activating patches
 
def max_activating_patches(opt, imgs, reps, 
        n_neurons = 5, n_max = 8, patch_size = 16, patch_spacing = 8, center_crop_margin = 16, do_savefig = False):
    # reps: b, c, x, y
    s = reps.shape

    neuron_inds = choice([i for i in range(s[1])], size = (n_neurons), replace=False)
    responses = reps[:, neuron_inds, :, :].permute(1, 0, 2, 3).reshape(n_neurons, -1) # n_neurons, b*x*y

    patches_list = []
    for n in range(n_neurons):
        inds_flattened = responses[n].argsort(descending=True) # b*x*y (sorted by value, largest first)
        inds = unravel_index(inds_flattened, (s[0], s[2], s[3])) # b*x*y, 3 (time, x, y)
        patches = inds[:n_max].clone() # n_max, 3 (max. activating patches)

        ctr = 0
        tns = []
        tns.append(patches[0][0])
        for np in range(1,n_max):
            tnp = patches[np][0]
            while tnp in tns:
                ctr += 1
                patches[np][:] = inds[n_max + ctr]
                tnp = patches[np][0]
            tns.append(tnp)
            

        patches_list.append(patches)

    plot_patches(opt, imgs, None, None, patches_list, n_neurons, n_max, 
        patch_size = patch_size, patch_spacing = patch_spacing, center_crop_margin = center_crop_margin, 
        plot_reference_patch = False, crop = True,  do_savefig = do_savefig)

    # embed()

##############################################################################################################

if __name__ == "__main__":

    opt = arg_parser.parse_args()

    model, _, test_loader = load_model_and_data(opt)

    imgs = test_loader.dataset.data
    class_names = test_loader.dataset.classes

    inputs, reps, targets = get_representations(opt, model, test_loader, reload = True)
    
    tSNE(opt, inputs, reps, targets, class_names) # , n_points = 1000)

    # patch_neighbouranalysis(opt, imgs, reps, n_examples = 5, n_neighbours = 8)

    max_activating_patches(opt, imgs, reps, n_neurons = 8, n_max = 10, do_savefig = True)

    embed()

    