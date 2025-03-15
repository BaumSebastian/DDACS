import yaml
import h5py
import numpy as np
from src import SimulationDataset, download_dataset


def main():
    config_file_path = r"./config/config.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    download = config["download_dataset"]
    data_dir = config["data_dir"]
    h5_subdir = config["h5_subdir"]
    url = config["dataset_url"]

    if download:
        download_dataset(url, data_dir)

    simulation_dataset = SimulationDataset(data_dir, h5_subdir)
    print(simulation_dataset)

    sim_id, metadata, h5_file_path = next(iter(simulation_dataset))
    print(
        "\n".join(
            [
                "Sample data entry.",
                f" - ID: {sim_id}",
                f" - Metadata: {metadata}",
                f" - h5 file path: {h5_file_path}",
            ]
        )
    )

    # Access the indvidual entry based on h5 structure.
    with h5py.File(h5_file_path, "r") as f:
        data = np.array(f["OP10"]["blank"]["node_displacement"])
    print(
        f"Example of pointcloud of 'blank' gemonetry for all ({data.shape[0]}) timesteps {data.shape}"
    )


if __name__ == "__main__":
    main()
