# Symmetry Preserving Attention Networks

A library for training and evaluation SPANets on jet reconstruction tasks. 
Originally developed for `ttbar` analysis,
this library now supports arbitrary event topologies and symmetry groups.

## Version 2 Update

We recently pushed an updated version 2 of this library which adds several new features.
1. New configuration file format with more options on inputs and event topology.
2. Allow for several different inputs, including global inputs for additional context.
3. New Regression and Classification output heads for performing per-event or per-particle predictions.
4. Gated transformers and linear layers for more robust networks. Less hyperparameter optimization.

An example demonstrating these new features may be found here: [docs/TTH.md](docs/TTH.md).

## Installation
You can install this package to use it outside of the repository after cloning.

```bash
git clone https://github.com/Alexanders101/SPANet
cd SPANet
pip install .
```

Alternatively, you can use `pip install -e .` to install in an editable mode.

### Note
The configuration `ini` has been deprecated into a configuration `yaml`. The conversion should just be to change the syntax a bit, the values should remain the same. A conversion script is in the works.

The dataset format has also changed slighty, but old style datasets can be converted into a new style dataset using [`utils/convert_dataset.py`](utils/convert_dataset.py).

The old codebase may always be accesed here: [https://github.com/Alexanders101/SPANet/tree/v1.0](https://github.com/Alexanders101/SPANet/tree/v1.0)

## Dependencies

A list of the libraries necessary to fully train and evaluate SPANets. 
These are only the minimum versions that we tested, other versions might work.

| Library                                                 | Minimum Version |
| ------------------------------------------------------- |-----------------|
| python                                                  | 3.9             |
| [numpy](https://pypi.org/project/numpy/)                | 1.22.1          |
| [sympy](https://www.sympy.org/en/index.html)            | 1.11            |
| [scikit_learn](https://scikit-learn.org/stable/)        | 1.1             |
| [pytorch](https://pytorch.org/)                         | 1.12            |
| [pytorch-lightning](https://www.pytorchlightning.ai/)   | 1.7             |
| [opt_einsum](https://github.com/dgasmith/opt_einsum)    | 3.3.0           |
| [h5py](https://pypi.org/project/h5py/)                  | 2.10            |
| [numba](https://numba.pydata.org/)                      | 0.53.1          |

We have updated to using an anaconda environment for simpler dependency management.
You can create the environment locally with the following conda / mamba commands:
```bash
conda env create -p ./environment --file environment.yaml
conda activate ./environment
```

## Example
We have provided a simple `ttbar` example in order to demonstrate how to
define events, construct datasets, and train & evaluate a network.


[Refer to this page for a detailed walk-through 
for the `ttbar` example](docs/TTBar.md).

The full `ttbar` dataset may be downloaded here: http://mlphysics.ics.uci.edu/data/2021_ttbar/.

We also have a more advanced example demonstrating some of the additinoal inputs and outputs available on a semi-leptonic `ttH` event. [Refer to this page for a detailed walk-through 
for the `ttH` example](docs/TTH.md).

## Usage
Using this library requires setting up several components. 
Refer to the following documentation pages in order to learn about the
the different setup components, or just follow the ttbar example.

1. [Defining the event topology](docs/EventInfo.md).
2. [Creating a training dataset](docs/Dataset.md).
3. [Configuring training options](docs/Options.md).


### Training

Once those steps are complete, you can begin training by 
calling `spanet.train` with your chosen parameters. For more information
simply run `python -m spanet.train --help`

You can experiment with the provided example configuration and dataset
for some `ttbar` events by calling 
`python -m spanet.train -of options_files/full_hadronic_ttbar/example.json --gpus NUM_GPUS` 
where `NUM_GPUS` is the number of gpus available on your machine.

### Evaluation

Once training is complete, you may evalute a network on
a testing dataset by running `spanet.test` with a path to your previously
trained network and a file on which to evalute on.

For example, after running the previous training run on `ttbar_example`, 
you can evaluate the network again on the example dataset by running.
`python -m spanet.test ./spanet_output/version_0 -tf data/full_hadronic_ttbar/example.h5`

Note that the included example file is very small and you will likely not
see very good performance on it.

## Citation
If you use this software for a publication, please cite the following:
```bibtex
@Article{10.21468/SciPostPhys.12.5.178,
	title={{SPANet: Generalized permutationless set assignment for particle physics  using symmetry preserving attention}},
	author={Alexander Shmakov and Michael James Fenton and Ta-Wei Ho and Shih-Chieh Hsu and Daniel Whiteson and Pierre Baldi},
	journal={SciPost Phys.},
	volume={12},
	pages={178},
	year={2022},
	publisher={SciPost},
	doi={10.21468/SciPostPhys.12.5.178},
	url={https://scipost.org/10.21468/SciPostPhys.12.5.178},
}
```
