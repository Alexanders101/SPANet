# SPANet Options

The SPANet options can be provided either programmatically or using a `.json` file.

## Programmatically Selecting Options
The `Options` class in [`spanet/options.py`](../spanet/options.py) defines all
of the valid options used during training and evaluation. You can quickly
experiment with different options by simply modifying this file or providing 
it as part of a different script.

## Options Files
[`spanet/options.py`](../spanet/options.py) defines the default parameters which
are fed to all networks. However, you often want to experiment with many different
options without having to always modify the codebase.

The other method of providing the options is by using a `.json` file
and a command line argument to `train.py`. An example of such a file is provided
in [`options_files/ttbar_example.json`](../options_files/ttbar_example.json). 

The `.json` options file will override all of the defaults specified in 
`spanet/options.py` for the given training run.
You can provide `train.py` an options file via the command line argument

`python train.py -of OPTIONS_FILE`

You can also programmatically load in a `.json` file with `Options.load(filepath)`
from [`spanet/options.py`](../spanet/options.py) if you are running your own
training script.

Whenever a network is training, it will always create a copy of its
current options in the checkpoint directory. 
The default output directories are of the form `spanet_output/version_*`.

## Command Line Arguments
`train.py` also allows you to temporarily override certain options using
command line arguments. You can view a complete list of these options
using `python train.py --help`. Some common ones include:

- `--gpus N` to select the number of GPUS to train on.
- `--batch_size N` to set the training batch_size.
- `--epochs N` to set the number of training epochs.
- `--log_dir DIR` to set the output directory to something different from the current directory.
- `--name NAME` to set the log directory name to something differnt than `lightning_logs`.