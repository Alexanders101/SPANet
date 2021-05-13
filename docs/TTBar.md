# `ttbar` Example Guide

Included in the repository are all files necessary to quickly test
SPANets on `ttbar` events. This repository only contains a tiny example
dataset for a sanity check, but you may acquire a larger training and testing data
set here: [TODO Upload Data]. Place the training and testing file into `data/ttbar` for
the examples to work correctly.

If you are using the docker container, then you can first alias `python` 
(or whatever command you want to call it) to the following docker command.

```
# GPU Enabled
# -----------
alias python="docker run --rm -it --gpus all -v $(pwd):/workspace --workdir=/workspace ashmakovuci/igb-python python"

# CPU Only
# --------
alias python="docker run --rm -it -v $(pwd):/workspace --workdir=/workspace ashmakovuci/igb-python python"
```

### Training
You can train a new SPANet on the ttbar dataset by running `train.py`.
Specifying `NUM_GPUS` will determine how many parallel GPUs the training process 
will use during training. You can set `NUM_GPUS` to be 0 to disable GPU training 
and only use the CPU. Make sure to continue the training until at least one 
epoch has finished so that it creates a checkpoint file.
Note that when using the full dataset,
a complete training run takes roughly 4 hours on a single GPU.

```
# Example Dataset
# ---------------
python train.py -of options_files/ttbar_example.json --gpus NUM_GPU

# Full Dataset
# ------------
python train.py -of options_files/ttbar_full.json --gpus NUM_GPUS
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

### Evaluation

Now we want to view the efficiency of the network on a testing dataset.
In the example config we are testing on the training dataset 
because it's the only dataset we have in the repo.
The full config will test on the proper testing dataset.
The following command will compute relevant performance metrics. 
The `--gpu` is optional. 
If you trained more than once, then adjust the version number accordingly.

```
# Example Dataset
# ---------------
python test.py ./spanet_output/version_0 -tf ./data/ttbar/ttbar_example.h5 --gpu

# Full Dataset
# ------------
python test.py ./spanet_output/version_0 -tf ./data/ttbar/ttbar_testing.h5 --gpu
```

Next we will output all SPANet predictions on a set of events
in order to analyze them further, simply run `predict.py` as follows.

```
# Example Dataset
# ---------------
python predict.py  ./spanet_output/version_0 ./spanet_ttbar_example_output.h5 -tf ./data/ttbar/ttbar_example.h5 --gpu`

# Full Dataset
# ------------
python predict.py  ./spanet_output/version_0 ./spanet_ttbar_testing_output.h5 -tf ./data/ttbar/ttbar_testing.h5 --gpu
```

This will create a new HDF5 File named `spanet_ttbar_output.h5` with the same
structure as the testing dataset except with the jet labels replaced
with the SPANet predictions. This can now be read in as a regular HDF5 
in order to perform other experiments.

## Event `.ini` File
We will now go over how the `ttbar` example configuration works.

Included is the event description for a full hadronic-decay `ttbar` event.
This file is located at [`event_files/ttbar.ini`](../event_files/ttbar.ini).
We will also reproduce it here for simplicity.

```
[SOURCE]
mass = log_normalize
pt = log_normalize
eta = normalize
phi = normalize
btag = none

[EVENT]
particles = (t1, t2)
permutations = [(t1, t2)]

[t1]
jets = (q1, q2, b)
permutations = [(q1, q2)]

[t2]
jets = (q1, q2, b)
permutations = [(q1, q2)]
```

### `[SOURCE]`
This section of the `.ini` file defines which features are present in our dataset.
You will notice that we define five unique features.
- `mass`
- `pt`
- `eta`
- `phi`
- `btag`

This is the information that we store for each jet. Also notice that we 
`log_normalize` the `mass` and `pt` features because they can have a large
range of possible values, normalize the `eta` and `phi` features for consistency,
and dont do anything with `btag` because it is already binary valued.

### `[EVENT]`
This section defines that our event has two particles which we are interested in
`t1` and `t2`. Furthermore, although in reality these particles should be a top 
quark and a top anti-quark, we are not differentiating the particles with respect 
to their charge. Therefore, we also note that `(t1, t2)` is a valid permutation to
apply to our targets, making the ordering of `t1` and `t2` arbitrary.

### Jet Definitions
```
[t1]
jets = (q1, q2, b)
permutations = [(q1, q2)]

[t2]
jets = (q1, q2, b)
permutations = [(q1, q2)]
```

These sections define that each top quark / anti-quark decay into three jets:
two light quarks `q1` and `q2` and a bottom quark `b`. Also notice that
we also tell SPANet that the particular ordering of `q1` and `q2` doesn't matter
similar to the `[EVENT]` section.

## `ttbar` Example Dataset
We provide in this repository a small example file which countains ~10,000
`ttbar` events to demonstrate the dataset structure. This file is located in
`data/ttbar/ttbar_example.h5`.

Examining the HDF5 file, it has the following structure
```
- source
---- mask
---- mass
---- pt
---- eta
---- phi
---- btag

- t1
---- mask
---- q1
---- q2
---- b

- t2
---- mask
---- q1
---- q2
---- b
```

You will notice that all the source feature names, the particle names, and 
the jet names correspond precisely to our event definition in `ttbar.ini`.

The example dataset contains `10,000` `ttbar` events with a maximum jet
multiplicity of `10`. The source features are float arrays of size 
`[10,000, 10]` because each jet has every feature assigned to it.
Not every event has all `10` jets, so any extra jets assigned to an event
are known as padding and all have feature values of 0. The `source/mask` array keeps track of which jets are real and which are
padding jets.

The particle groups contain the indices of each jet associated with each particle.
Each array in these groups is an integer array of size `[10,000]`. The
`t1/mask` and `t2/mask` arrays determine if the particle is fully reconstructable
in the given event.