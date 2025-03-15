from darus import Dataset as DarusDataset


def download_dataset(dataset_url: str, data_dir: str) -> None:
    """
    Download a dataset from a given URL and save it to a local directory.

    :param dataset_url: The url to the dataset on DaRUS.
    :type dataset_url: str
    :param data_dir: The root directory to save the dataset.
    :type data_dir: str
    """
    ds = DarusDataset(dataset_url)
    ds.summary()
    ds.download(data_dir)
