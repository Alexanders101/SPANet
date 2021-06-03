# Symmetry Preserving Attention Networks

A library for training and evaluation SPANets on jet reconstruction tasks. 
Originally developed for `ttbar` analysis,
this library now supports arbitrary event topologies and symmetry groups.

## Dependencies

A list of the libraries necessary to fully train and evaluate SPANets. 
These are only the minimum versions that we tested, other versions might work.

| Library                                                 | Minimum Version |
| ------------------------------------------------------- | --------------- |
| python                                                  | 3.8             |
| [numpy](https://pypi.org/project/numpy/)                | 1.19.5          |
| [sympy](https://www.sympy.org/en/index.html)            | 1.8             |
| [scikit_learn](https://scikit-learn.org/stable/)        | 0.24.2          |
| [pytorch](https://pytorch.org/)                         | 1.8.1           |
| [pytorch-lightning](https://www.pytorchlightning.ai/)   | 1.2.10          |
| [opt_einsum](https://github.com/dgasmith/opt_einsum)    | 3.3.0           |
| [h5py](https://pypi.org/project/h5py/)                  | 2.10            |
| [numba](https://numba.pydata.org/)                      | 0.53.1          |

A docker container with python and all of these libraries already installed
is available here: https://hub.docker.com/r/spanet/spanet-python

## Example
We have provided a simple `ttbar` example in order to demonstrate how to
define events, construct datasets, and train & evaluate a network.


[Refer to this page for a detailed walk-through 
for the `ttbar` example](docs/TTBar.md).

The full `ttbar` dataset may be downloaded here: http://mlphysics.ics.uci.edu/data/2021_ttbar/.

## Usage
Using this library requires setting up several components. 
Refer to the following documentation pages in order to learn about the
the different setup components, or just follow the ttbar example.

1. [Defining the event topology](docs/EventInfo.md).
2. [Creating a training dataset](docs/Dataset.md).
3. [Configuring training options](docs/Options.md).


### Training

Once those steps are complete, you can begin training by 
calling `train.py` with your chosen parameters. For more information
simply run `python train.py --help`

You can experiment with the provided example configuration and dataset
for some `ttbar` events by calling 
`python train.py -of options_files/ttbar_example.json --gpus NUM_GPUS` 
where `NUM_GPUS` is the number of gpus available on your machine.

### Evaluation

Once training is complete, you may evalute a network on
a testing dataset by running `test.py` with a path to your previously
trained network and a file on which to evalute on.

For example, after running the previous training run on `ttbar_example`, 
you can evaluate the network again on the example dataset by running.
`python test.py ./spanet_output/version_0 -tf ./data/ttbar/ttbar_example.h5`

Note that the included example file is very small and you will likely not
see very good performance on it.
