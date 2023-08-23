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
| [numpy](https://pypi.org/project/numpy/)                | 1.24            |
| [sympy](https://www.sympy.org/en/index.html)            | 1.11            |
| [scikit_learn](https://scikit-learn.org/stable/)        | 1.1             |
| [pytorch](https://pytorch.org/)                         | 2.0             |
| [pytorch-lightning](https://www.pytorchlightning.ai/)   | 2.0             |
| [opt_einsum](https://github.com/dgasmith/opt_einsum)    | 3.3.0           |
| [h5py](https://pypi.org/project/h5py/)                  | 2.10            |
| [numba](https://numba.pydata.org/)                      | 0.56            |

We have updated to using an anaconda environment for simpler dependency management.
You can create the environment locally with the following conda / mamba commands:
```bash
conda env create -p ./environment --file environment_cuda118.yaml
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

### Exporting

Once you are happy with your model, you can export it to an [ONNX](https://onnxruntime.ai/) file to use in external applications. This can be done by running `spanet.export` with the log directory and the desired output file. For example: `python -m spanet.export ./spanet_output/version_0 spanet.onnx`.

Note that only the neural network is able to be exported, and this network outputs the full reconstruction distributions for every event. Unfortunately, the reconstruction algorithm defined [here](spanet/network/prediction_selection.py) cannot be exported as part of the ONNX graph. If your target application uses python, then you can simply use SPANet's selection algorithm, but non-python applications must define their own selection algorithm.

You may examine all of the inputs and outputs with the following snippet:
```python
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime

session = onnxruntime.InferenceSession(
    "./spanet.onnx", 
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print("Inputs:", [input.name for input in session.get_inputs()])
print("Outputs:", [output.name for output in session.get_outputs()])
```

#### Inputs

| Input                     | Shape       | DType |
|---------------------------|-------------|-------|
| {sequential_input_1}_data | (B, N1, D1) | float |
| {sequential_input_1}_mask | (B, N1)     | bool  |
| {sequential_input_2}_data | (B, N2, D2) | float |
| {sequential_input_2}_mask | (B, N2)     | bool  |
| {global_input_1}_data     | (B, 1, D1)  | float |
| {global_input_1}_mask     | (B, 1)      | bool  |
| {global_input_2}_data     | (B, 1, D2)  | float |
| {global_input_2}_mask     | (B, 1)      | bool  |

The ONNX model expects two inputs for every `INPUT` defined in the event file. Replace the values in the braces with their appropriate names. The data contains the features for each input. The features must be provided in the **exact order** that they are defined in the event file. Notice that global inputs require a dummy axis to be added to match the overall shape of the sequential inputs. 

**Log Features:** Any features marked either `log` or `log_normalize` must have the following preprocessing transformation applied `f(x) -> log(x + 1)`. You can skip this log preprocessing and have it performed by the network if you specify `--input-log-transform`. However, this operation is expensive to perform by the graph, so we recommend you apply it during your data pipeline for maximum efficiency.

#### Outputs
| Output                                    | Shape          | DType |
|-------------------------------------------|----------------|-------|
| {event_particle_1}_assignment_probability | (B, N, N, ...) | float |
| {event_particle_2}_assignment_probability | (B, N, N, ...) | float |
| {event_particle_1}_detection_probability  | (B)            | float |
| {event_particle_2}_detection_probability  | (B)            | float |
| {regression_target_1}                     | (B)            | float |
| {regression_target_2}                     | (B)            | float |
| {classification_target_1}                 | (B, C)         | float |
| {classification_target_2}                 | (B, C)         | float |

The ONNX model may produce any of the valid output heads. Each event partile defined produces an assignment distribution for its reconstruction. This distribution with be a singlet/doublet/triplet/etc. joint distribution depending on the number of decay products defined for each particle. The shape will reflect this number of products. For example, if a particle has two decay products, then its `assignment_log_probability` will have a shape of `(B, N, N)`. Each particle also has associated with it a `detection probability` which indicates how likely the particle is to be reconstructable.

The additional outputs will only be present if you define any `REGRESSION` or `CLASSIFICATION` outputs in the event file. Each of the definitions will be add an extra output. The regression outputs simply contain the predicted value for each regression target. The classification outputs contain a distribution over possible classes for each target.

**Log Probability vs. Probability** For additional numerical stability, you may choose to output the log distributions, `log P(x)`, for all probability outputs instead. If you specify `--output-log-transform` in the export script, then the `*_assignment_probability` and `*_detection_probability` outputs will be replaced with `*_assignment_log_probability` and `*_detection_log_probability`. The classification outputs will also be represented as log-probabilities, although the name will not change.

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
