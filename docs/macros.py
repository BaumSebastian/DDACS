"""mkdocs-macros entry point.

This module is documentation-only. Display constants live at the top so that:
  * the value travels with the git tag (RTD shows the right number per version)
  * no runtime file I/O is required at build time
  * everything stays in one place, not coupled into the package config
"""

from __future__ import annotations

from pathlib import Path

from ddacs import croissant as _croissant
from ddacs.config import METADATA_FILE, SMALL_TEST_FILES

# ---------------------------------------------------------------------------
# Display constants — bump these once per release.
# ---------------------------------------------------------------------------
SIMULATION_COUNT = 32_466
SIMULATION_COUNT_CORNER = 32_070
SIMULATION_COUNT_RDDAC = 396

TOTAL_SIZE = "~640 GB"  # full dataset on DaRUS, all 22 files
PER_SIM_SIZE = "~20 MB"  # one HDF5 file after gzip + shuffle
SMALL_DOWNLOAD_SIZE = "~22 MB"  # `ddacs download --small` (sample + CSV + manifest)

# ---------------------------------------------------------------------------
# Croissant manifest helpers (read at build time, not coupled to the values
# above — they describe the schema, not file sizes / counts).
#
# Preference order:
#   1. docs/metadata.json   — bundled with the docs build; what RTD uses.
#                             Version-locked to the git tag; refresh on each
#                             dataset release: cp <data_dir>/metadata.json docs/.
#   2. data/metadata.json   — local developer copy (the data/ symlink); kept
#                             for backwards compatibility so existing dev
#                             workflows keep working.
#   3. DaRUS URL            — last-resort network fetch; flaky on RTD because
#                             mlcroissant's URL-load path can raise
#                             "AssertionError: Found no node in graph".
# ---------------------------------------------------------------------------
_DOCS_METADATA = Path(__file__).resolve().parent / METADATA_FILE
_LOCAL_METADATA = Path(__file__).resolve().parent.parent / "data" / METADATA_FILE


def _dataset():
    for candidate in (_DOCS_METADATA, _LOCAL_METADATA):
        if candidate.is_file():
            return _croissant.load(source=candidate)
    return _croissant.load(source=None)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{line}\n{sep}\n{body}"


def define_env(env):
    """Plug-in entry point — mkdocs-macros calls this at build time."""

    # ----- simple value substitutions -----
    @env.macro
    def simulation_count() -> str:
        return f"{SIMULATION_COUNT:,}"

    @env.macro
    def simulation_count_corner() -> str:
        return f"{SIMULATION_COUNT_CORNER:,}"

    @env.macro
    def simulation_count_rddac() -> str:
        return f"{SIMULATION_COUNT_RDDAC:,}"

    @env.macro
    def total_size() -> str:
        return TOTAL_SIZE

    @env.macro
    def per_sim_size() -> str:
        return PER_SIM_SIZE

    @env.macro
    def small_download_size() -> str:
        return SMALL_DOWNLOAD_SIZE

    # ----- small auto-built table summarising the counts -----
    @env.macro
    def simulation_stats() -> str:
        return _md_table(
            ["", "Count"],
            [
                ["DDACS (total)", f"{SIMULATION_COUNT:,}"],
                ["RDDAC sub study", f"{SIMULATION_COUNT_RDDAC:,}"],
            ],
        )

    # ----- file lists sourced from ddacs.config -----
    @env.macro
    def small_test_files() -> str:
        """Space-separated list used by `ddacs download --small`."""
        return " ".join(SMALL_TEST_FILES)

    @env.macro
    def metadata_url() -> str:
        """Permanent DaRUS download URL for `metadata.json`."""
        return _croissant.METADATA_URL

    # ----- schema tables sourced from metadata.json -----
    @env.macro
    def process_parameters_table() -> str:
        ds = _dataset()
        descs = _croissant.process_parameters_descriptions(ds)
        rows = [[f"`{name}`", desc] for name, desc in descs.items()]
        return _md_table(["Column", "Description"], rows)

    @env.macro
    def hdf5_field_table(prefix: str = "") -> str:
        ds = _dataset()
        fields = _croissant.field_map(ds)
        rows = []
        for name, f in fields.items():
            desc = (f.description or "").replace("\n", " ")
            if prefix and prefix not in desc:
                continue
            rows.append([f"`{name}`", desc[:120] + ("…" if len(desc) > 120 else "")])
        if not rows:
            return f"_no HDF5 fields matching prefix `{prefix}` found_"
        return _md_table(["Field", "Description"], rows)
