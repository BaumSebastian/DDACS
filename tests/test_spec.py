"""Tests for the dataset specification (ddacs.spec)."""

from __future__ import annotations

import dataclasses

import pytest

import ddacs
from ddacs.spec import DDACS_SAMPLE_SIM_ID, DDACS_SPEC, _ddacs_default_version


class TestDatasetSpec:
    def test_ddacs_spec_identity(self):
        assert DDACS_SPEC.name == "DDACS" and DDACS_SPEC.prog == "ddacs"
        assert DDACS_SPEC.dataset_doi == "doi:10.18419/DARUS-4801"
        assert DDACS_SPEC.id_format.format(DDACS_SAMPLE_SIM_ID) == "258864"
        assert f"{DDACS_SAMPLE_SIM_ID}.zip" in DDACS_SPEC.small_test_files
        assert DDACS_SPEC.metadata_file in DDACS_SPEC.small_test_files

    def test_default_version_tracks_package_major(self):
        major = ddacs.__version__.split(".")[0]
        assert _ddacs_default_version() == f"{major}.0"
        assert DDACS_SPEC.default_version == f"{major}.0"

    def test_spec_is_immutable_and_reusable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DDACS_SPEC.name = "other"  # type: ignore[misc]
        sibling = dataclasses.replace(DDACS_SPEC, name="RDDAC", prog="rddac", id_format="{:04d}")
        assert sibling.id_format.format(42) == "0042"
        assert sibling.darus_base_url == DDACS_SPEC.darus_base_url
