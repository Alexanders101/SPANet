# `ttbar` Example Guide

## Environment
We include an [anaconda environment](./environment.yml) file which can be used to install the required dependencies. You can get the anaconda package manager at [this link](https://www.anaconda.com/products/distribution). Alternatively, you can use the [mamba package manager](https://mamba.readthedocs.io/en/latest/installation.html). The environment can be installed with the following command:
```bash
conda env create -p ./environment --file environment.yaml
conda activate ./environment
```

## Installation
Optionally, you can install this package to use it outside of the git directory.

Either run `pip install .` or, for an editable install, `pip install -e .` from the root directory of the repository.

## Full Training Data


Included in the repository are all files necessary to quickly test SPANets on `ttbar` events. This repository only contains a tiny example dataset for a sanity check, but you may acquire a larger training and testing data set here: [http://mlphysics.ics.uci.edu/data/2021_ttbar/](http://mlphysics.ics.uci.edu/data/2021_ttbar/). As of 2022/10/20, this dataset is still in the old style to ensure backwards compatibility. You may run the following commands to download and convert the dataset (In the SPANet root directory).
```bash
wget -O ./data/full_hadronic_ttbar/training.h5 http://mlphysics.ics.uci.edu/data/2021_ttbar/ttbar_training.h5
python utils/convert_dataset.py ./data/full_hadronic_ttbar/training.h5 ./data/full_hadronic_ttbar/training.h5
````

### Testing Data
```bash
wget -O ./data/full_hadronic_ttbar/testing.h5 http://mlphysics.ics.uci.edu/data/2021_ttbar/ttbar_testing.h5
python utils/convert_dataset.py ./data/full_hadronic_ttbar/testing.h5 ./data/full_hadronic_ttbar/testing.h5
````

## Training
You can train a new SPANet on the ttbar dataset by running `train.py`.
Specifying `NUM_GPUS` will determine how many parallel GPUs the training process 
will use during training. You can set `NUM_GPUS` to be 0 to disable GPU training 
and only use the CPU. Make sure to continue the training until at least one 
epoch has finished so that it creates a checkpoint file.
Note that when using the full dataset,
a complete training run takes roughly 4 hours on a single GPU.

```bash
# Example Dataset
# ---------------
python -m spanet.train -of options_files/full_hadronic_ttbar/example.json --time_limit 00:00:01:00 --gpus NUM_GPU

# Full Dataset
# ------------
python -m spanet.train -of options_files/full_hadronic_ttbar/full_training.json --gpus NUM_GPUS
```

If you get a `RuntimeError: CUDA out of memory` then you need to decrease the
batch size so that the data can fit onto your GPU. You can achieve this by adding
`--batch_size BATCH_SIZE` to the `train.py` command above. Reasonable values include
512, 256, or 64. (The larger the better as long as it fits in memory.)

Training results, logs, and weights will be stored in `./spanet_output` by default
.Every time you start a new training, a new version will be created in the output directory. 
If you want to load from a checkpoint, 
output to a different location, 
or change something else about training,
simply run `python train.py --help` for a full list of options and how to set them.

During training, we output `tensorboard` logs to track performance. 
To view these logs, have `tensorboard` installed (included in the docker container)
and run

`tensorbard --logdir spanet_output`

Afterwards, navigate to `localhost:6006` in a browser.

## Evaluation

Now we want to view the efficiency of the network on a testing dataset.
In the example config we are testing on the training dataset 
because it's the only dataset we have in the repo.
The full config will test on the proper testing dataset.
The following command will compute relevant performance metrics. 
The `--gpu` is optional. 
If you trained more than once, then adjust the version number accordingly.

```bash
# Example Dataset
# ---------------
python -m spanet.test ./spanet_output/version_0 -tf data/full_hadronic_ttbar/example.h5 --gpu

# Full Dataset
# ------------
python -m spanet.test ./spanet_output/version_0 -tf data/full_hadronic_ttbar/testing.h5 --gpu
```

Next we will output all SPANet predictions on a set of events
in order to analyze them further, simply run `predict.py` as follows.

```bash
# Example Dataset
# ---------------
python -m spanet.predict ./spanet_output/version_0 ./spanet_ttbar_example_output.h5 -tf data/full_hadronic_ttbar/example.h5 --gpu

# Full Dataset
# ------------
python -m spanet.predict ./spanet_output/version_0 ./spanet_ttbar_testing_output.h5 -tf data/full_hadronic_ttbar/testing.h5 --gpu
```

This will create a new HDF5 File named `spanet_ttbar_output.h5` with the same
structure as the testing dataset except with the jet labels replaced
with the SPANet predictions. This can now be read in as a regular HDF5 
in order to perform other experiments.

## Event `.yaml` File
We will now go over how the `ttbar` example configuration works.

Included is the event description for a full hadronic-decay `ttbar` event.
This file is located at [`event_files/full_hadronic_ttbar.yaml`](../event_files/full_hadronic_ttbar.yaml).
We will also reproduce it here for simplicity. Any `CAPITAL_CASE` keys are special keys that cannot be used by any other element of the tree. They must always be `CAPITAL_CASE`. 

```yaml
# ---------------------------------------------------
# REQUIRED - INPUTS - List all inputs to SPANet here.
# ---------------------------------------------------
INPUTS:
  # -----------------------------------------------------------------------------
  # REQUIRED - SEQUENTIAL - inputs which can have an arbitrary number of vectors.
  # -----------------------------------------------------------------------------
  SEQUENTIAL:
    Source:
      mass: log_normalize
      pt: log_normalize
      eta: normalize
      phi: normalize
      btag: none

  # ---------------------------------------------------------------------
  # REQUIRED - GLOBAL - inputs which will have a single vector per event.
  # ---------------------------------------------------------------------
  GLOBAL:


# ----------------------------------------------------------------------
# REQUIRED - EVENT - Complete list of resonance particles and daughters.
# ----------------------------------------------------------------------
EVENT:
  t1:
    - q1
    - q2
    - b
  t2:
    - q1
    - q2
    - b

# ---------------------------------------------------------
# REQUIRED KEY - PERMUTATIONS - List of valid permutations.
# ---------------------------------------------------------
PERMUTATIONS:
    EVENT:
      - [ t1, t2 ]
    t1:
      - [ q1, q2 ]
    t2:
      - [ q1, q2 ]


# ------------------------------------------------------------------------------
# REQUIRED - REGRESSIONS - List of desired features to regress from observables.
# ------------------------------------------------------------------------------
REGRESSIONS:


# -----------------------------------------------------------------------------
# REQUIRED - REGRESSIONS - List of desired classes to predict from observables.
# -----------------------------------------------------------------------------
CLASSIFICATIONS:
```

### `INPUTS`
The Inputs section of the `.yaml` file defines which features are present in our dataset which the network will use to make predictions.

```yaml
SEQUENTIAL:
    Source:
```
This defines a sequential (variable length) input named `Source`. This will contain the observable features for our hadronic jets. You will notice that we define five unique features.
- `mass`
- `pt`
- `eta`
- `phi`
- `btag`

This is the information that we store for each jet. Also notice that we 
`log_normalize` the `mass` and `pt` features because they can have a large
range of possible values, normalize the `eta` and `phi` features for consistency,
and dont do anything with `btag` because it is already binary valued.

### `EVENT`
This section defines that Feynman diagram structure of our event. We have two **event particles** which we are interested in `t1` and `t2`. Each top quark / anti-quark decay into three jets: two light quarks `q1` and `q2` and a bottom quark `b`. These final jets are observable and so they belong in the second stage of diagram. We currently only support depth two Feynman diagrams. The first stage should be the particles we are interested in studying, and the second the observable decay products.

### `PERMUTATIONS`
This is where we may define invariant symmetries for our assignment reconstruction.

Although in reality the event particles should be a top quark and a top anti-quark, we are not differentiating the particles with respect to their charge. Therefore, we set `(t1, t2)` to be a valid permutation of the event particles, making the ordering of `t1` and `t2` arbitrary. We also tell SPANet that the particular ordering of `q1` and `q2` jets within each top doesn't matter with an additional `(q1, q2)` permutation for each top quark.

### `REGRESSIONS` and `CLASSIFICATIONS`
This section allows us to define custom per-event, per-particle, and per-decay product regressions and classifications. This is still an experimental feature and not used for `ttbar`.


## `ttbar` Example Dataset
We provide in this repository a small example file which countains ~10,000
`ttbar` events to demonstrate the dataset structure. This file is located in
`data/full_hadronic_ttbar/example.h5`.

You can example HDF5 file structure with the following command: ``
```bash
$ python utils/examine_hdf5.py data/full_hadronic_ttbar/example.h5 --shape

============================================================
| Structure: data/full_hadronic_ttbar/example.h5
============================================================

|-INPUTS
|---Source
|-----MASK                       :: bool     : (10000, 10)
|-----btag                       :: float32  : (10000, 10)
|-----eta                        :: float32  : (10000, 10)
|-----mass                       :: float32  : (10000, 10)
|-----phi                        :: float32  : (10000, 10)
|-----pt                         :: float32  : (10000, 10)
|-TARGETS
|---t1
|-----b                          :: int64    : (10000,)
|-----q1                         :: int64    : (10000,)
|-----q2                         :: int64    : (10000,)
|---t2
|-----b                          :: int64    : (10000,)
|-----q1                         :: int64    : (10000,)
|-----q2                         :: int64    : (10000,)
```

You will notice that all the source feature names, the event particle names, and 
the decay product names correspond precisely to our event definition in `full_hadronic_ttbar.yaml`.

The example dataset contains `10,000` `ttbar` events with a maximum jet
multiplicity of `10`. The source features are float arrays of size 
`[10,000, 10]` because each jet has every feature assigned to it.
Not every event has all `10` jets, so any extra jets assigned to an event
are known as padding and all have feature values of 0. The `INPUTS/Source/MASK` array keeps track of which jets are real and which are
padding jets.

The particle groups contain the indices of each jet associated with each particle.
Each array in these groups is an integer array of size `[10,000]`. Any jets which are missing from the event are marked with a `-1` value in the corresponding target array.
