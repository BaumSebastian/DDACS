"""Dataset specification — the knobs that distinguish one DaRUS dataset from another.

The ddacs machinery (croissant, streaming, pytorch, h5_tools, cli) is generic
over any DaRUS-hosted, Croissant-described HDF5 dataset. Everything
dataset-specific lives in a :class:`DatasetSpec`; every machinery function
accepts an optional ``spec`` keyword defaulting to :data:`DDACS_SPEC`, so
existing DDACS code keeps working unchanged while sibling datasets (e.g. the
experimental RDDAC counterpart) reuse the machinery by passing their own spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version


@dataclass(frozen=True)
class DatasetSpec:
    """Identity and layout of one DaRUS-hosted Croissant/HDF5 dataset."""

    # Display name, used in CLI panels and messages (e.g. "DDACS").
    name: str
    # CLI program name (e.g. "ddacs").
    prog: str
    # DaRUS persistent identifier, e.g. "doi:10.18419/DARUS-4801".
    dataset_doi: str
    # Dataset version the CLI downloads by default (e.g. "3.0").
    default_version: str
    # DaRUS instance base URL.
    darus_base_url: str = "https://darus.uni-stuttgart.de"

    # Canonical file names inside the dataset.
    process_parameters_file: str = "process_parameters.csv"
    metadata_file: str = "metadata.json"

    # Column in process_parameters that holds the record id, and how that id
    # maps to the HDF5 member name inside the zips ("{}" -> "258864.h5",
    # "{:04d}" -> "0042.h5").
    id_column: str = "index"
    id_format: str = "{}"

    # Files fetched by `<prog> download --small`.
    small_test_files: tuple[str, ...] = ()

    # Default local data directory (CLI --out and load(data_dir=...)).
    default_data_dir: str = "./data"

    # Croissant manifest @ids / defaults.
    field_map_record_set: str = "field-map"
    default_field_data_type: str = "sc:Float"


def _ddacs_default_version() -> str:
    """Dataset version tied to the installed package major: 3.x.y -> "3.0"."""
    try:
        return f"{_pkg_version('ddacs').split('.')[0]}.0"
    except PackageNotFoundError:  # editable/source checkout without metadata
        return "3.0"


DDACS_SAMPLE_SIM_ID = 258864

DDACS_SPEC = DatasetSpec(
    name="DDACS",
    prog="ddacs",
    dataset_doi="doi:10.18419/DARUS-4801",
    default_version=_ddacs_default_version(),
    small_test_files=(
        "process_parameters.csv",
        "metadata.json",
        f"{DDACS_SAMPLE_SIM_ID}.zip",
    ),
)
