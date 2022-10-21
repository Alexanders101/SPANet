from typing import Optional, List
from tqdm import tqdm

import h5py
import numpy as np


def structure_printer(file, shape: bool = True, indent=0):
    space = 32 - 2 * indent - 2
    if isinstance(file, h5py.Dataset):
        if shape:
            print(f" :: {str(file.dtype):8} : {file.shape}", end="")

        return

    for key in file:
        print("\n|-" + indent * "--" + key.ljust(space), end="")
        structure_printer(file[key], shape, indent + 1)


def write(input_file, output_file, path: Optional[List[str]] = None, verbose: bool = True):
    if path is None:
        path = []

    for key, value in input_file.items():
        current_subpath = path + [key]
        if isinstance(value, np.ndarray):
            if verbose:
                print(f"Creating {'/'.join(current_subpath)}: Shape {value.shape}")
            output_file.create_dataset("/".join(current_subpath), data=value)
        else:
            write(input_file[key], output_file, current_subpath, verbose=verbose)


def load_dataset(dataset):
    values = dataset[:]
    if values.dtype == np.float64:
        values = values.astype(np.float32)

    if values.dtype == np.int32:
        values = values.astype(np.int64)

    return values


def read(file, level=0, path=[]):
    if isinstance(file, h5py.Dataset):
        return load_dataset(file)

    database = {}

    iterator = file
    if level == 1:
        iterator = tqdm(file, f"Loading {path[-1]}")

    for key in iterator:
        database[key] = read(file[key], level + 1, path + [key])

    return database


def extract(file):
    if isinstance(file, h5py.Dataset):
        return file[:]

    database = {}
    for key in file:
        database[key] = extract(file[key])

    return database


def concatenate(head, *tail, path: Optional[List[str]] = None):
    if path is None:
        path = []

    if not isinstance(head, dict):
        return np.concatenate((head, *tail))

    database = {}
    for key in head:
        new_path = path + [key]
        print(f"Concatenating: {'/'.join(new_path)}")
        try:
            database[key] = concatenate(head[key], *[d[key] for d in tail], path=new_path)
        except KeyError:
            print(f"Skipping: {'/'.join(new_path)}")
            continue

    return database
