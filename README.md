
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5593214.svg)](https://doi.org/10.5281/zenodo.5593214)
<!-- Should be updated with new release! Please check -->

# CLAPP code

This is the code for the publication:

B. Illing, J. Ventura, G. Bellec & W. Gerstner
[*Local plasticity rules can learn deep representations using self-supervised contrastive predictions*](https://arxiv.org/abs/2010.08262), accepted at NeurIPS 2021.

Contact:
[bernd.illing@epfl.ch](mailto:bernd.illing@epfl.ch)

# Implementation of CLAPP in pytorch

We implement CLAPP (and its variants) using the auto-differentiation provided by [`pytorch`](https://pytorch.org). That means that we do not implement the learning rule, Equations (6) - (8), explicitely. Instead, we apply the CLAPP loss, Equation (3), at every layer and block gradients (pytorch `.detach()`), such that the automatically calculated gradients (`.backward()`) match the CLAPP learning rules. We summarize for a single layer in `python/pytorch` pseudocode:

```python
""" 
require:
layer (encoder layer to train)
clapp_hinge_loss (CLAPP hinge loss as in Equation (3); contains the prediction weights)
opt (optimiser, e.g. ADAM, containing all trainable parameters of this layer)
x_past (previous input)
x (current input)
"""

c = layer(x_past.detach()) # context: encoding of previous input
z = layer(x.detach()) # future activity

loss = clapp_hinge_loss(c, z)
loss.backward() # autodiff calculates gradients

opt.step() # update parameters of layer and prediction weights
```

We verified numerically that the obtained updates are equivalent to evaluating the CLAPP learning rules Equations (6) - (8). The code for this can be found in `./vision/CLAPPVision/vision/compare_updates.py`, see Vision section for more details.

Note that for Hinge Loss CPC, the end-to-end version of CLAPP, we only use a single CLAPP loss at the final layer. Furthermore, we don't use the `.detach()` function to allow gradient flow through the whole network. 

Variants of CLAPP mainly differ in the exact implementation of the CLAPP loss `clapp_hinge_loss`. E.g. for the synchronous version CLAPP-s, the CLAPP loss adds the contribution of negative and positive sample at every step, instead of sampling with 50/50 probability as in CLAPP.


# Structure of the code

The code is divided into three independent sections, corresponding to the three domains we apply CLAPP to:

* vision
* video
* audio

Each section comes with its own dependencies handled by `conda` environments, as explained in the respective sections below.

# Vision

The implementation of the CLAPP vision experiments is based on Sindy Löwe's code of the [Greedy InfoMax model](https://github.com/loeweX/Greedy_InfoMax).

## Setup

To setup the conda environment, simply run

```bash
    cd vision
    bash ./setup_dependencies.sh
```

To activate and deactive the created conda environment, run

```bash
    conda activate clappvision
    conda deactivate
```

respectively. 

## Usage

We included three sample scripts to run CLAPP, CLAPP-s (synchronous pos. and neg. updates; version with symmetric pre- and retrodiction weights) and Hinge Loss CPC (end-to-end version of CLAPP). To run the, e.g. the Hinge Loss CPC simulations (model training + evaluation), run:

```bash
    cd vision
    bash ./scripts/vision_traineval_HingeLossCPC.sh
```

The code includes many (experimental) versions of CLAPP as command line options that are not used and mentioned in the paper. To view all command-line options of model training, run:

```bash
    cd vision
    python -m CLAPPVision.vision.main_vision --help
```

We also added code to run the above mentioned numerical check that the updates obtained with auto-differentiation are equivalent to evaluating the CLAPP learning rules. To check this, e.g. for a randomly initialised network at the first epoch of training, run:

```bash
    mkdir ./logs/CLAPP_init/
    python -m CLAPPVision.vision.compare_updates --download_dataset --save_dir CLAPP_init --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --contrast_mode 'hinge' --num_epochs 1 --negative_samples 1 --sample_negs_locally --sample_negs_locally_same_everywhere --start_epoch 0 --model_path ./logs/CLAPP_init/ --save_vars_for_update_calc 3 --batch_size 4
```

The equivalence was found to also hold later during training. For this, the respective simulations first need to be run (see comments in `./vision/CLAPPVision/vision/compare_updates.py`). 

# Video

The implementation of the CLAPP video experiments was inspired by Tengda Han's code for [Dense Predictive Coding](https://github.com/TengdaHan/DPC)

## Setup

The setup of the conda environment is described in `./video/env_setup.txt`. To activate and deactive the created conda environment `pdm`, run

```bash
    conda activate pdm
    conda deactivate
```

respectively.

## Usage

The basic simulations described in the paper can be replicated using the commands listed in `./video/commands.txt`.


# Audio

The implementation of the CLAPP audio experiments is based on Sindy Löwe's code of the [Greedy InfoMax model](https://github.com/loeweX/Greedy_InfoMax).

<!-- GUILLAUME: Your instructions go here. Please publish release after updating; this should trigger zenodo to update the link of the DOI -->

## Setup

## Usage

# Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{illing2020local,
  title={Local plasticity rules can learn deep representations using self-supervised contrastive predictions},
  author={Illing, Bernd and Ventura, Jean and Bellec, Guillaume and Gerstner, Wulfram},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {34},
  year={2021}
}
```
