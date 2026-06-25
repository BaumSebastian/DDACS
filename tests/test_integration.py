"""Integration tests against a real `ddacs download --small` payload.

Skipped automatically if the download can't succeed (no token, no network,
draft requires auth and `DARUS_API_TOKEN` is unset, …). When the public
v3.0 ships these will run token-free.
"""

from __future__ import annotations

import ddacs


def test_load_against_real_manifest(small_data_dir):
    ds = ddacs.load(data_dir=str(small_data_dir))
    assert ds.metadata.name == "DDACS"
    # We always ship process-parameters + field-map + several views.
    rs_ids = {rs.id for rs in ds.metadata.record_sets}
    assert {"process-parameters", "field-map", "springback-minimal"} <= rs_ids


def test_mapping_picks_up_small_files(small_data_dir):
    ds = ddacs.load(data_dir=str(small_data_dir))
    # Expect the sample zip (258864.zip) and the CSV at minimum.
    names = {n.split("/")[-1] for n in (ds.mapping or {}).values()}
    assert "258864.zip" in names
    assert "process_parameters.csv" in names


def test_open_h5_on_sample_sim(small_data_dir):
    with ddacs.open_h5(258864, data_dir=str(small_data_dir)) as f:
        assert "OP10/blank/node_displacement" in f
        assert int(f.attrs["index"]) == 258864


def test_inspect_h5_runs(small_data_dir, capsys):
    with ddacs.open_h5(258864, data_dir=str(small_data_dir)) as f:
        ddacs.inspect_h5(f)
    out = capsys.readouterr().out
    assert "OP10/" in out
