import h5py

from argparse import ArgumentParser
from glob import glob

from shared import concatenate, write, extract


def main(input_folder: str, output_file: str):
    print("=" * 40)
    print(f"Reading in files from {input_folder}")
    print("-" * 40)
    global_file = []
    for filename in glob(f"{input_folder}/*.h5"):
        print(f"Reading: {filename}")
        with h5py.File(filename, 'r') as file:
            global_file.append(extract(file))
    print("=" * 40)
    print()

    print("=" * 40)
    print("Concatenating Files")
    print("-" * 40)
    global_file = concatenate(*global_file)
    print("=" * 40)
    print()

    print("=" * 40)
    print(f"Writing Output to {output_file}")
    print("-" * 40)
    with h5py.File(output_file, 'w') as output_file:
        write(global_file, output_file)
    print("=" * 40)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_folder", type=str, help="Folder of HDF5 files to concatenate.")
    parser.add_argument("output_file", type=str, help="Complete HDF5 file to create for output.")

    args = parser.parse_args()
    main(args.input_folder, args.output_file)
