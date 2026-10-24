import json
import shutil

import pytest
from release_factory import make_release

import tokamaksi as tk
from tokamaksi import hf


def test_open_by_path_and_check_clean(synthetic):
    rel = tk.open_release(synthetic)
    assert rel.tag == "v1" and rel.schema == 1
    assert rel.check(deep=True) == []
    assert tk.open_release(synthetic) is rel
    assert tk.open_release(rel) is rel


def test_flipped_byte_is_reported(tmp_path, synthetic):
    copy = tmp_path / "copy"
    shutil.copytree(synthetic, copy)
    victim = copy / "labels" / "reviewed" / "train.parquet"
    data = bytearray(victim.read_bytes())
    data[100] ^= 0xFF
    victim.write_bytes(bytes(data))
    problems = tk.open_release(copy).check(deep=True)
    assert any("labels/reviewed/train.parquet" in p and "sha256" in p for p in problems)
    victim.write_bytes(bytes(data)[:-5])
    assert any("bytes" in p for p in tk.open_release(copy).check())


def test_missing_required_file(tmp_path, synthetic):
    copy = tmp_path / "copy"
    shutil.copytree(synthetic, copy)
    (copy / "taxonomy.yaml").unlink()
    assert any("taxonomy.yaml" in p for p in tk.open_release(copy).check())


def test_a_stray_file_is_reported(tmp_path, synthetic):
    copy = tmp_path / "copy"
    shutil.copytree(synthetic, copy)
    (copy / "labels" / "reviewed" / "extra.parquet").write_bytes(b"x")
    (copy / "labels" / ".hidden").write_bytes(b"x")
    problems = tk.open_release(copy).check()
    assert problems == ["labels/reviewed/extra.parquet is not listed in release.json"]


def test_not_a_release(tmp_path):
    with pytest.raises(tk.ReleaseError, match="release.json"):
        tk.open_release(tmp_path)


def test_newer_schema_is_refused_and_a_missing_key_reads_as_one(tmp_path):
    newer = make_release(tmp_path / "newer", schema=2)
    with pytest.raises(tk.SchemaError, match="Upgrade"):
        tk.open_release(newer)
    assert tk.open_release(make_release(tmp_path / "one", schema=1)).schema == 1
    manifest = json.loads((tmp_path / "one" / "release.json").read_text())
    assert "schema" in manifest and "schema" not in json.loads(
        (make_release(tmp_path / "none") / "release.json").read_text()
    )


def test_tag_resolves_under_home_without_the_network(monkeypatch, tmp_path, synthetic):
    shutil.copytree(synthetic, hf.release_dir("v7"))
    monkeypatch.setattr(
        hf, "hub_backend", lambda *a, **k: pytest.fail("touched the network")
    )
    assert tk.open_release("v7").path == hf.release_dir("v7").resolve()


def test_missing_tag_offline_names_the_command():
    with pytest.raises(tk.OfflineError, match="tokamaksi pull v3"):
        tk.open_release("v3")


def test_missing_tag_without_pull_never_downloads(monkeypatch):
    monkeypatch.delenv("TOKAMAKSI_OFFLINE")
    monkeypatch.setattr(
        hf, "hub_backend", lambda *a, **k: pytest.fail("touched the network")
    )
    with pytest.raises(tk.OfflineError, match="pull=False"):
        tk.open_release("v3", pull=False)


def test_datasets_models_card(rel):
    assert set(rel.datasets()["tier"]) == {"reviewed", "original"}
    assert list(rel.datasets(event="confinement")["tier"]) == ["original"]
    assert rel.models()["edge_localized_mode"]["task"] == "detector"
    assert rel.card().startswith("# Synthetic")
