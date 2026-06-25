"""Single source of truth for DDACS package constants.

File names, URLs and dataset identifiers used by the package live here so a
rename or migration only touches one file.
"""

from importlib.metadata import version as _pkg_version

# DaRUS hosting
DARUS_BASE_URL = "https://darus.uni-stuttgart.de"
DATASET_DOI = "doi:10.18419/DARUS-4801"

# Default dataset version downloaded by `ddacs download`. Tied to the package
# major: ddacs 3.x.y -> dataset version "3.0".
DEFAULT_VERSION = f"{_pkg_version(__package__).split('.')[0]}.0"

# Canonical file names inside the dataset (post-extraction)
PROCESS_PARAMETERS_FILE = "process_parameters.csv"
METADATA_FILE = "metadata.json"
H5_SUBDIR = "h5"

# Column that holds the simulation ID (matches the HDF5 filename)
ID_COLUMN = "index"

# Single-simulation sample for fast preview + small-test downloads
SAMPLE_SIM_ID = 258864
SAMPLE_ZIP_FILE = f"{SAMPLE_SIM_ID}.zip"

# Files downloaded by `ddacs download --small`
SMALL_TEST_FILES = [PROCESS_PARAMETERS_FILE, METADATA_FILE, SAMPLE_ZIP_FILE]

# Default output directory used by `ddacs download --out` and
# `ddacs.load(data_dir=...)`. Kept symmetric so files written by the CLI are
# picked up by `load()` without an extra argument.
DEFAULT_DATA_DIR = "./data"

# Croissant manifest @ids the code references when composing or reading the
# manifest. Kept here so a manifest rename only touches one file.
FIELD_MAP_RECORD_SET = "field-map"

# Fallback dataType for fields created via `ddacs.add_view` when the source
# field-map entry doesn't carry one. Croissant 1.1 uses schema.org datatypes.
DEFAULT_FIELD_DATA_TYPE = "sc:Float"
