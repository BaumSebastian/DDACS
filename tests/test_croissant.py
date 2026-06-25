"""Tests for `ddacs.load`, `ddacs.add_view` and internal parsers."""

from __future__ import annotations

import pytest

import ddacs
from ddacs.croissant import _normalize_field_spec, _slicing_to_jsonpath

# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------


class TestNormalizeFieldSpec:
    def test_bare_string_means_whole_field(self):
        assert _normalize_field_spec("foo") == ("foo", None)

    def test_tuple_explicit_none(self):
        assert _normalize_field_spec(("foo", None)) == ("foo", None)

    def test_tuple_integer(self):
        assert _normalize_field_spec(("foo", 2)) == ("foo", 2)

    def test_tuple_list(self):
        assert _normalize_field_spec(("foo", [1, 2, 3])) == ("foo", [1, 2, 3])

    def test_rejects_wrong_tuple_arity(self):
        with pytest.raises(TypeError):
            _normalize_field_spec(("foo",))

    def test_rejects_non_string_field_id(self):
        with pytest.raises(TypeError):
            _normalize_field_spec((42, 0))


class TestSlicingToJsonPath:
    def test_none_is_no_transform(self):
        assert _slicing_to_jsonpath(None) is None

    def test_integer_single_index(self):
        assert _slicing_to_jsonpath(2) == "$[2]"

    def test_list_multi_index(self):
        assert _slicing_to_jsonpath([2, 3]) == "$[2,3]"

    def test_rejects_bool(self):
        # bool is an int subclass; we don't want True/False to silently work.
        with pytest.raises(TypeError):
            _slicing_to_jsonpath(True)

    def test_rejects_mixed_list(self):
        with pytest.raises(TypeError):
            _slicing_to_jsonpath([1, "x"])


# ---------------------------------------------------------------------------
# `load` over the synthetic dataset
# ---------------------------------------------------------------------------


class TestLoadSynthetic:
    def test_returns_mlc_dataset(self, synthetic_data_dir):
        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        assert type(ds).__module__.startswith("mlcroissant")
        assert ds.metadata.name == "synthetic"

    def test_mapping_picks_up_local_files(self, synthetic_data_dir):
        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        # 3 zips + 1 csv
        assert len(ds.mapping or {}) == 4

    def test_record_sets_parsed(self, synthetic_data_dir):
        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        ids = {rs.id for rs in ds.metadata.record_sets}
        assert ids == {"process-parameters", "field-map", "springback-minimal"}

    def test_file_set_contained_in_parsed(self, synthetic_data_dir):
        """Regression check: `cr:containedIn` alias in the manifest's @context."""
        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        fs = next(n for n in ds.metadata.distribution if type(n).__name__ == "FileSet")
        assert len(fs.contained_in) == 3


# ---------------------------------------------------------------------------
# `add_view`
# ---------------------------------------------------------------------------


class TestAddView:
    def test_adds_record_set_in_place(self, synthetic_data_dir):
        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        before = {rs.id for rs in ds.metadata.record_sets}
        ddacs.add_view(
            ds,
            "my-view",
            fields={
                "forming": ("op10_blank_node_displacement", 2),
                "thickness": "op10_blank_element_shell_thickness",
            },
        )
        after = {rs.id for rs in ds.metadata.record_sets}
        assert after == before | {"my-view"}

    def test_field_specs_round_trip(self, synthetic_data_dir):
        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        ddacs.add_view(
            ds,
            "my-view",
            fields={
                "whole": "op10_blank_node_displacement",
                "one": ("op10_blank_node_displacement", 2),
                "subset": ("op10_blank_node_displacement", [2, 3]),
            },
        )
        rs = next(r for r in ds.metadata.record_sets if r.id == "my-view")
        by_name = {f.name: f for f in rs.fields}

        # "whole" should have no transform
        assert not by_name["whole"].source.transforms

        # "one" should have $[2]
        assert by_name["one"].source.transforms[0].json_path == "$[2]"

        # "subset" should have $[2,3]
        assert by_name["subset"].source.transforms[0].json_path == "$[2,3]"

    def test_returns_dataset_for_chaining(self, synthetic_data_dir):
        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        out = ddacs.add_view(ds, "tmp", fields={"x": "op10_blank_node_displacement"})
        assert out is ds
