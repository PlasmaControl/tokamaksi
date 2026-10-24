import json
import shutil

import pytest

import tokamaksi as tk
from tokamaksi import hf


@pytest.fixture
def online(monkeypatch):
    monkeypatch.delenv("TOKAMAKSI_OFFLINE")


def _fake(synthetic, calls=None):
    def backend(repo, tag, dest, patterns):
        if calls is not None:
            calls.append((repo, tag, patterns))
        shutil.copytree(synthetic, dest, dirs_exist_ok=True)
        (dest / ".cache" / "huggingface").mkdir(parents=True)
        (dest / ".cache" / "huggingface" / "x.metadata").write_text("x")
        return "c0ffee"

    return backend


def test_pull_installs_atomically_and_cleans_up(online, synthetic):
    calls = []
    target = hf.pull("v1", backend=_fake(synthetic, calls))
    assert target == hf.release_dir("v1")
    assert calls[0][0] == "nc1/tokamak-si" and "labels/**" in calls[0][2]
    assert "tables/**" not in calls[0][2]
    assert not (target / ".cache").exists()
    stamp = json.loads((target / ".tokamaksi.json").read_text())
    assert stamp["commit"] == "c0ffee" and stamp["tag"] == "v1"
    assert [p.name for p in hf.releases_dir().iterdir()] == ["v1"]  # no partial left
    assert tk.open_release("v1").commit == "c0ffee"
    assert target.stat().st_mode & 0o777 == 0o755
    assert (target / "release.json").stat().st_mode & 0o777 == 0o644


def test_open_release_pulls_a_missing_tag(online, monkeypatch, synthetic):
    monkeypatch.setattr(hf, "hub_backend", _fake(synthetic))
    assert tk.open_release("v1").tag == "v1"


def test_second_pull_is_a_no_op_unless_forced(online, synthetic):
    hf.pull("v1", backend=_fake(synthetic))
    hf.pull("v1", backend=lambda *a: pytest.fail("downloaded again"))
    calls = []
    hf.pull("v1", force=True, backend=_fake(synthetic, calls))
    assert len(calls) == 1


def test_checksum_mismatch_leaves_nothing(online, synthetic):
    def corrupt(repo, tag, dest, patterns):
        _fake(synthetic)(repo, tag, dest, patterns)
        (dest / "grid" / "gold.parquet").write_bytes(b"not parquet")

    with pytest.raises(tk.ReleaseError, match="grid/gold.parquet"):
        hf.pull("v1", backend=corrupt)
    assert list(hf.releases_dir().iterdir()) == []


def test_a_racing_pull_keeps_the_first(online, synthetic):
    def racer(repo, tag, dest, patterns):
        _fake(synthetic)(repo, tag, dest, patterns)
        hf.release_dir(tag).mkdir(parents=True)  # somebody else got there first
        (hf.release_dir(tag) / "theirs").write_text("x")

    target = hf.pull("v1", backend=racer)
    assert (target / "theirs").exists()
    assert [p.name for p in hf.releases_dir().iterdir()] == ["v1"]


def test_offline_pull_says_so_and_a_tag_on_disk_still_answers(synthetic):
    with pytest.raises(tk.OfflineError, match="network is off"):
        hf.pull("v1", backend=_fake(synthetic))
    shutil.copytree(synthetic, hf.release_dir("v1"))
    assert hf.pull(
        "v1", backend=lambda *a: pytest.fail("downloaded")
    ) == hf.release_dir("v1")


@pytest.mark.parametrize("tag", ["../x", "main", "v1/../v2", ""])
def test_a_tag_must_look_like_one(online, tag, synthetic):
    with pytest.raises(tk.ReleaseError, match="release tag"):
        hf.pull(tag, backend=_fake(synthetic))


def test_access_error_names_three_steps_and_no_token():
    error = hf._access_error("nc1/tokamak-si", "v1", "HTTP 401")
    text = str(error)
    assert "hf auth login" in text and "HF_TOKEN" in text and "read access" in text
    assert "hf_" not in text.replace("HF_TOKEN", "")
