import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

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
        shutil.copytree(synthetic, hf.release_dir(tag))  # somebody else got there first
        (hf.release_dir(tag) / "theirs").write_text("x")

    target = hf.pull("v1", backend=racer)
    assert (target / "theirs").exists()
    assert [p.name for p in hf.releases_dir().iterdir()] == ["v1"]


def test_a_racer_that_leaves_no_release_json_is_refused(online, synthetic):
    def racer(repo, tag, dest, patterns):
        _fake(synthetic)(repo, tag, dest, patterns)
        hf.release_dir(tag).mkdir(parents=True)
        (hf.release_dir(tag) / "theirs").write_text("x")  # not a release

    with pytest.raises(tk.ReleaseError, match="--force"):
        hf.pull("v1", backend=racer)
    assert [p.name for p in hf.releases_dir().iterdir()] == ["v1"]  # no partial left
    assert (hf.release_dir("v1") / "theirs").exists()


def test_an_empty_leftover_folder_is_replaced(online, synthetic):
    hf.release_dir("v1").mkdir(parents=True)  # a failed earlier run, or a mkdir
    calls = []
    target = hf.pull("v1", backend=_fake(synthetic, calls))
    assert (target / "release.json").is_file() and len(calls) == 1
    assert [p.name for p in hf.releases_dir().iterdir()] == ["v1"]
    hf.pull("v1", backend=lambda *a: pytest.fail("downloaded again"))
    assert tk.open_release("v1").commit == "c0ffee"


def test_open_release_downloads_once_over_an_empty_folder(
    online, monkeypatch, synthetic
):
    hf.release_dir("v1").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(hf, "hub_backend", _fake(synthetic, calls))
    assert tk.open_release("v1").tag == "v1"
    tk.release._OPEN.clear()  # a new process: nothing remembered
    assert tk.open_release("v1").tag == "v1" and len(calls) == 1


def test_a_folder_that_is_not_a_release_is_never_deleted_without_force(
    online, synthetic
):
    stray = hf.release_dir("v1")
    stray.mkdir(parents=True)
    (stray / "mine.txt").write_text("keep me")
    with pytest.raises(tk.ReleaseError, match="no release.json.*--force"):
        hf.pull("v1", backend=lambda *a: pytest.fail("downloaded for nothing"))
    with pytest.raises(tk.ReleaseError, match="not a release"):
        tk.open_release(stray)
    assert (stray / "mine.txt").read_text() == "keep me"
    hf.pull("v1", force=True, backend=_fake(synthetic))
    assert (stray / "release.json").is_file() and not (stray / "mine.txt").exists()
    assert [p.name for p in hf.releases_dir().iterdir()] == ["v1"]  # nothing set aside


def test_force_puts_the_old_copy_back_when_it_cannot_install(
    online, monkeypatch, synthetic
):
    hf.pull("v1", backend=_fake(synthetic))
    (hf.release_dir("v1") / "old-marker").write_text("x")
    real = os.replace

    def refuse(src, dst, *a, **k):
        if Path(src).name.startswith(".partial-"):
            raise PermissionError("no space")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(tk.ReleaseError, match="could not install"):
        hf.pull("v1", force=True, backend=_fake(synthetic))
    monkeypatch.setattr(os, "replace", real)
    assert (hf.release_dir("v1") / "old-marker").exists()
    assert [p.name for p in hf.releases_dir().iterdir()] == ["v1"]


def test_the_stamp_names_the_tokamaksi_version(online, synthetic):
    target = hf.pull("v1", backend=_fake(synthetic))
    stamp = json.loads((target / ".tokamaksi.json").read_text())
    assert stamp["tokamaksi_version"] == tk.__version__


@pytest.mark.parametrize(
    "module, name", [("httpx", "ConnectError"), ("requests", "ConnectionError")]
)
def test_a_dead_network_is_an_offline_error(online, monkeypatch, module, name):
    import huggingface_hub

    error = getattr(pytest.importorskip(module), name)("no route to host")

    def unreachable(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", unreachable)
    with pytest.raises(tk.OfflineError, match="cannot reach the Hub.*no route"):
        hf.hub_backend("nc1/tokamak-si", "v1", Path("unused"), ("labels/**",))


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


@pytest.mark.parametrize(
    "status, kind",
    [(401, tk.AccessError), (404, tk.AccessError), (503, tk.OfflineError)],
)
def test_hub_answers_map_to_our_errors(online, monkeypatch, status, kind):
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    error = HfHubHTTPError.__new__(HfHubHTTPError)  # its constructor differs by version
    error.response = SimpleNamespace(status_code=status)

    def answer(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", answer)
    with pytest.raises(kind, match=str(status)):
        hf.hub_backend("nc1/tokamak-si", "v1", Path("unused"), ("labels/**",))


def test_access_error_names_three_steps_and_no_token():
    error = hf._access_error("nc1/tokamak-si", "v1", "HTTP 401")
    text = str(error)
    assert "hf auth login" in text and "HF_TOKEN" in text and "read access" in text
    assert "hf_" not in text.replace("HF_TOKEN", "")
