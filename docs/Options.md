# SPANet Options

The SPANet options can be provided either programmatically or using a `.json` file.

## Programmatically Selecting Options
The `Options` class in [`spanet/options.py`](../spanet/options.py) defines all
of the valid options used during training and evaluation. You can quickly
experiment with different options by simply modifying this file or providing 
it as part of a different script.

## Options Files
The other method of provoding the options for a run is using a `.json` file
and a command line argument to `train.py`. An example of such a file is provided
in [`options_files/ttbar_example.json`](../options_files/ttbar_example.json). 

You can give `train.py` an options file to use using the command line argument

`python train.py -of OPTIONS_FILE`

Whenever a network is training, it will always create a copy of its
current options in the checkpoint directory. 
Default is `spanet_output/version_*`.

## Command Line Arguments
`train.py` also allows you to temporarily override certain options using
command line arguments. You can view a complete list of these options
using `python train.py --help`. Some common ones include:

- `--gpus N` to select the number of GPUS to train on.
- `--batch_size N` to set the training batch_size.
- `--epochs N` to set the number of training epochs.
- `--log_dir DIR` to set the output directory to something different from the current directory.
- `--name NAME` to set the log directory name to something differnt than `lightning_logs`.

## Complete Options Listing
TODO