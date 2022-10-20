import h5py

from argparse import ArgumentParser
from shared import structure_printer


def main(filepath: str):
    with h5py.File(filepath, 'r') as file:
        print("Structure")
        print("=" * 40)
        structure_printer(file)
        print("\n")

        # print("Target Prevalence")
        # print("=" * 40)
        # for side in sorted(file['target']):
        #     for target in sorted(file['target'][side]):
        #         if target == "mask":
        #             continue
        #
        #         pretty_side = side.split("_")[0].capitalize()
        #         pretty_name = target.capitalize()
        #         quark = " Quark" if len(pretty_name) <= 2 else ""
        #
        #         pretty_title = f"{pretty_side} {pretty_name}{quark}"
        #         pretty_value = f"{100 * (file['target'][side][target][:] >= 0).mean():.2f}"
        #         print(f"{pretty_title.ljust(16)}:  {pretty_value.rjust(5)}%")
        #
        #     print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str, help="HDF5 file to examine")

    main(parser.parse_args().filepath)
