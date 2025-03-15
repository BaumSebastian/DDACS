import yaml
from src import BaseFEMDataset, PunchDataset, download_dataset


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

    id, p, X = next(iter(bds))
    print("\nSampel data entry.\n" + f"ID: {id}\n" + f"Metadata: {p}\n" + f"Data: {X}")

    pds = PunchDataset(root, data_dir)

    y, (p, X) = next(iter(pds))
    print(
        "\nSampel data entry.\n"
        + f"Label: {y}\n"
        + f"Metadata: {p}\n"
        + f"Point Cloud Data Shape: {X.shape}"
    )


if __name__ == "__main__":
    main()
