import h5py

from argparse import ArgumentParser

from shared import structure_printer, read, write


def convert_dataset(dataset):
    # Convert input features
    # ======================
    inputs = {
        "Source": {
            key: value
            for key, value in dataset["source"].items()
            if key != "mask"
        }
    }

    inputs["Source"]["MASK"] = dataset["source"]["mask"]

    # Convert assignment targets
    # ==========================
    targets = {
        event_particle: {
            product_particle: targets
            for product_particle, targets in product_particles.items()
            if product_particle != "mask"
        }
        for event_particle, product_particles in dataset.items()
        if event_particle != "source"
    }

    return {
        "INPUTS": inputs,
        "TARGETS": targets
    }


def main(input_filepath: str, output_filepath: str):
    print()
    print("=" * 40)
    print("Input file structure")
    print("=" * 40)
    with h5py.File(input_filepath, 'r') as input_file:
        structure_printer(input_file)
        print("\n")

        dataset = read(input_file)

    print()
    print("=" * 40)
    print("Converting Dataset")
    print("=" * 40)
    new_dataset = convert_dataset(dataset)

    print()
    print("=" * 40)
    print("Creating output file")
    print("=" * 40)
    with h5py.File(output_filepath, 'w') as output_file:
        write(new_dataset, output_file)

    print()
    print("=" * 40)
    print("Output file structure")
    print("=" * 40)
    with h5py.File(output_filepath, 'r') as output_file:
        structure_printer(output_file)
        print("\n")

    pass
if __name__ == "__main__":
    parser = ArgumentParser(description="Convert old V1 HDF5 datasets into the new format.")
    parser.add_argument("input_filepath", type=str, help="Input V1 HDF5 file.")
    parser.add_argument("output_filepath", type=str, help="Output V2 HDF5 file.")

    arguments = parser.parse_args()
    main(arguments.input_filepath, arguments.output_filepath)
