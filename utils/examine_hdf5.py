import h5py

from argparse import ArgumentParser
from shared import structure_printer


def main(filepath: str, shape: bool = False):
    with h5py.File(filepath, 'r') as file:
        print("=" * 60)
        print(f"| Structure for {filepath} ")
        print("=" * 60)
        structure_printer(file, shape)
        print("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str, help="HDF5 file to examine")
    parser.add_argument("-s", "--shape", action="store_true", help="Print the shape of each leaf node.")

    arguments = parser.parse_args()
    main(arguments.filepath, arguments.shape)
