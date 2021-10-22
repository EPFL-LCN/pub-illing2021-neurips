
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5593214.svg)](https://doi.org/10.5281/zenodo.5593214)
<!-- Should be updated with new release! Please check -->

This is the code for the publication:

B. Illing, J. Ventura, G. Bellec & W. Gerstner
[*Local plasticity rules can learn deep representations using self-supervised contrastive predictions*](https://arxiv.org/abs/2010.08262), accepted at NeurIPS 2021.

Contact:
[bernd.illing@epfl.ch](mailto:bernd.illing@epfl.ch)

# Structure of the code

The code is divided into three independent sections, corresponding to the three applications we apply CLAPP to:

* vision
* video
* audio

Each section comes with its own dependencies handled by conda environments, as explained in the respective sections below.

# Vision

The implementation of the CLAPP vision experiments is based on Sindy Löwe's code of the [Greedy InfoMax model](https://github.com/loeweX/Greedy_InfoMax).

## Setup

To setup the conda environment, simply run

```bash
    bash ./vision/setup_dependencies.sh
```

To activate and deactive the created conda environment, run

```bash
    conda activate infomax
    conda deactivate
```

respectively. The environment name `infomax`, as well as the name of our python module `GreedyInfoMax`, are GIM code legacy. 

## Usage

We included three sample scripts to run CLAPP, CLAPP-s (synchronous pos. and neg. updates; version with weight symmetry in $W^{pred}$) and Hinge Loss CPC (end-to-end version of CLAPP). To run the, e.g. the Hinge Loss CPC simulations (model training + evaluation), run:

```bash
    bash ./vision/scripts/vision_traineval_HingeLossCPC.sh
```

The code includes many (experimental) versions of CLAPP as command line options that are not used and mentioned in the paper. To view all command-line options of model training, run:

```bash
    cd vision
    python -m GreedyInfoMax.vision.main_vision --help
```

Training in general uses auto-differentiation provided by `pytorch`. We checked that the obtained updates are equivalent to evaluating the CLAPP learning rules for $W$ and $W^{pred}$, Equations (6) - (8). The used code for this sanity check can be found in `./vision/GreedyInfoMax/vision/compare_updates.py`.


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
@article{illing2020local,
  title={Local plasticity rules can learn deep representations using self-supervised contrastive predictions},
  author={Illing, Bernd and Ventura, Jean and Bellec, Guillaume and Gerstner, Wulfram},
  journal={arXiv preprint arXiv:2010.08262},
  year={2020}
}
```
