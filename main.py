import yaml
import h5py
import numpy as np
from src import BaseFEMDataset, download_dataset

def main():

    config_file_path = r"./config/config_template.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    download = config["download_dataset"]
    root = config["root"]
    data_dir = config["data_dir"]
    url = config["url"]

    if download:
        download_dataset(url, root)
    
    bds = BaseFEMDataset(root, data_dir)
    print(bds)

    sim_id, metadata, h5_file_path = next(iter(bds))
    print("\n".join(["Sampel data entry.", f" - ID: {sim_id}", f" - Metadata: {metadata}", f" - h5 file path: {h5_file_path}"]))

    # Access the indvidual entry based on h5 structure.
    with h5py.File(h5_file_path, "r") as f:
            data = np.array(f["OP10"]["blank"]["node_displacement"])
    print(f"Example of pointcloud of 'blank' gemonetry for all (4) timesteps {data.shape}")

if __name__ == "__main__":
    main()