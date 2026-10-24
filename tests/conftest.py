import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from release_factory import make_corpus, make_release  # noqa: E402

import tokamaksi  # noqa: E402
from tokamaksi import labels, release  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """No test reads or writes the user's own home, release cache or Hub settings."""
    for name in (
        "TOKAMAKSI_HOME",
        "TOKAMAKSI_RELEASE",
        "TOKAMAKSI_REPO",
        "TOKAMAKSI_OFFLINE",
        "TOKAMAKSI_CORPUS",
        "HF_HUB_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TOKAMAKSI_HOME", str(tmp_path / "home"))
    # Offline unless a test turns it on with a fake backend: no test can reach the Hub.
    monkeypatch.setenv("TOKAMAKSI_OFFLINE", "1")
    labels._WARNED.clear()
    release._OPEN.clear()
    yield


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory) -> Path:
    """One synthetic release for the session; tests must not modify it."""
    return make_release(tmp_path_factory.mktemp("release") / "v1")


@pytest.fixture
def rel(synthetic) -> "tokamaksi.Release":
    return tokamaksi.open_release(synthetic)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("corpus")
    make_corpus(folder)
    return folder
