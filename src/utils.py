from darus import Dataset as DarusDataset


def download_dataset(url: str, root: str) -> None:
    """
    Download a dataset from a given URL and save it to a local directory.

    :param url: The url to the dataset on DaRUS.
    :type url: str
    :param root: The local directory to save the dataset.
    :type root: str
    """
    ds = DarusDataset(url)
    ds.summary()
    ds.download(root)
