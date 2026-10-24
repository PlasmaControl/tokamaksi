"""Nothing that must stay private is tracked: no data, no cluster paths, no tokens.

What it checks: no data-file suffixes, no file over 1 MB, none of the file names a
release is made of, no cluster path, Hub token or ELM challenge key in any tracked
text, and a `.gitignore` that blocks data without blocking package code. What it
does not check: names of reviewers or third-party tables, and the challenge's shot
list; the maintainer's pre-push search covers those.

The patterns are assembled from pieces so this file does not trip its own search.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="needs a git checkout",
)

DATA_SUFFIXES = {
    ".parquet",
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    ".pkl",
    ".pt",
    ".ckpt",
    ".jsonl",
    ".csv",
}
#: The files a release is made of: a real copy of one must never be committed.
RELEASE_PARTS = {
    "release.json",
    "taxonomy.yaml",
    "signals.yaml",
    "datasets.parquet",
    "LICENSE.md",
    ".tokamaksi.json",
}
FORBIDDEN = {
    "a cluster path": re.compile(
        "|".join(
            (
                "/scr" + "atch/gpfs",
                "/proj" + "ects/",
                "/ho" + "me/",
                "EKOLE" + "MEN",
                "nc15" + "14",
                "gat" + ".com",
            )
        )
    ),
    "a Hub token": re.compile("hf" + "_[A-Za-z0-9]{30,}"),
    "the ELM challenge key": re.compile(
        "|".join(
            (
                "ELM_data" + "base",
                "answer" + "_key",
                "elm_challenge" + "_key",
                "b71f" + "991",
            )
        )
    ),
}


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [name for name in out.stdout.split("\0") if name and (ROOT / name).is_file()]


def test_no_data_files_or_big_files_are_tracked():
    names = _tracked()
    assert names, "git lists no files"
    data = [n for n in names if Path(n).suffix.lower() in DATA_SUFFIXES]
    assert not data, f"data files tracked: {data}"
    big = [n for n in names if (ROOT / n).stat().st_size > 1_000_000]
    assert not big, f"files over 1 MB tracked: {big}"
    parts = [n for n in names if Path(n).name in RELEASE_PARTS]
    assert not parts, f"release files tracked: {parts}"


def test_no_private_strings_in_tracked_files():
    found = []
    for name in _tracked():
        text = (ROOT / name).read_text(encoding="utf-8", errors="ignore")
        for what, pattern in FORBIDDEN.items():
            if pattern.search(text):
                found.append(f"{name}: {what}")
    assert not found, found


@pytest.mark.parametrize(
    "path, ignored",
    [
        ("labels/reviewed/train.parquet", True),
        ("shot.h5", True),
        ("signals.npz", True),
        ("labels.csv", True),
        ("data/anything.yaml", True),
        ("releases/v1/release.json", True),
        ("src/tokamaksi/plot.py", False),
        ("src/tokamaksi/data/signal_catalog.yaml", False),
        ("tests/test_labels.py", False),
        ("docs/schema.md", False),
    ],
)
def test_gitignore_blocks_data_but_not_package_code(path, ignored):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert (result.returncode == 0) is ignored
