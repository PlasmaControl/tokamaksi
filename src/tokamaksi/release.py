"""A release: a directory of plain files in the layout of the Hub dataset."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from . import config
from .errors import OfflineError, ReleaseError, SchemaError
from .schema import (
    RELEASE_SCHEMA_MAX,
    REQUIRED_DIRS,
    REQUIRED_FILES,
    SHOT_COLUMNS,
    require_columns,
)
from .taxonomy import Taxonomy

#: Releases opened in this process, by resolved path.
_OPEN: dict[Path, Release] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Release:
    """One release directory, read lazily.

    `path` holds `release.json`, `taxonomy.yaml`, `datasets.parquet`, `labels/`,
    `grid/` and `shots/`. Every table is read on demand and none is written.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path).expanduser().resolve()
        manifest = self.path / "release.json"
        if not manifest.is_file():
            raise ReleaseError(f"{self.path} is not a release: it has no release.json")
        try:
            self.meta: dict = json.loads(manifest.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ReleaseError(f"{manifest} is not valid JSON: {error}") from error
        self.tag: str = str(self.meta.get("release") or self.path.name)
        #: release.json has no `schema` key until the writer adds one: that is 1.
        self.schema: int = int(self.meta.get("schema", 1))
        if self.schema > RELEASE_SCHEMA_MAX:
            raise SchemaError(
                f"release {self.tag} has schema {self.schema}; this tokamaksi reads up "
                f"to {RELEASE_SCHEMA_MAX}. Upgrade tokamaksi."
            )
        self.commit: str | None = None
        stamp = self.path / ".tokamaksi.json"
        if stamp.is_file():
            try:
                self.commit = json.loads(stamp.read_text(encoding="utf-8")).get(
                    "commit"
                )
            except ValueError:
                pass
        self._taxonomy: Taxonomy | None = None

    def __repr__(self) -> str:
        return f"Release({self.tag!r}, {str(self.path)!r})"

    def taxonomy(self) -> Taxonomy:
        """The release's event taxonomy."""
        if self._taxonomy is None:
            self._taxonomy = Taxonomy.from_file(self.path / "taxonomy.yaml")
        return self._taxonomy

    def datasets(
        self, event: str | None = None, tier: str | None = None
    ) -> pd.DataFrame:
        """`datasets.parquet`: one row per label table (dataset, event, tier, layer,
        method, version, n_rows, n_shots, categories, shipped, ...). Start here to
        see which tiers and methods cover an event."""
        frame = pd.read_parquet(self.path / "datasets.parquet")
        if event is not None:
            frame = frame[frame["event"] == event]
        if tier is not None:
            frame = frame[frame["tier"] == tier]
        return frame.reset_index(drop=True)

    def labels(self, event=None, **kwargs) -> pd.DataFrame:
        """Same as `load_labels(event, release=self, **kwargs)`."""
        from .labels import load_labels

        return load_labels(event, release=self, **kwargs)

    def grid(self, which: str = "gold"):
        """Same as `load_grid(which, release=self)`."""
        from .grid import load_grid

        return load_grid(which, release=self)

    def shots(self, which: str) -> pd.DataFrame:
        """A shot list: `gold`, `silver`, `holdout` or `cohort_split`."""
        from .errors import MissingPartError

        if which not in SHOT_COLUMNS:
            raise ValueError(
                f"no shot list {which!r}; the lists are {sorted(SHOT_COLUMNS)}"
            )
        path = self.path / "shots" / f"{which}.parquet"
        if not path.is_file():
            have = sorted(p.stem for p in (self.path / "shots").glob("*.parquet"))
            raise MissingPartError(
                f"release {self.tag} has no {which} shot list; it has {', '.join(have)}"
            )
        frame = pq.read_table(path).to_pandas()
        require_columns(frame.columns, SHOT_COLUMNS[which], f"shots/{which}.parquet")
        return frame

    def models(self) -> dict:
        """The release's `signals.yaml`: per event, the best model, the signals it
        reads and its score. It is not a catalogue of every signal."""
        path = self.path / "signals.yaml"
        if not path.is_file():
            return {}
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def card(self) -> str:
        """The dataset card (`README.md`), or an empty string."""
        path = self.path / "README.md"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def check(self, *, deep: bool = False) -> list[str]:
        """Problems found, as sentences (an empty list is fine); never raises.

        Sizes of every present listed file are compared; `deep=True` also compares
        sha256. A file listed under `tables/` may be absent (it is not fetched).
        """
        problems = []
        for name in REQUIRED_FILES:
            if not (self.path / name).is_file():
                problems.append(f"missing required file {name}")
        for name in REQUIRED_DIRS:
            if not (self.path / name).is_dir():
                problems.append(f"missing required folder {name}/")
        listed = self.meta.get("files")
        if not isinstance(listed, dict):
            return problems
        for name, info in sorted(listed.items()):
            path = self.path / name
            if not path.is_file():
                if not name.startswith("tables/"):
                    problems.append(f"listed file {name} is missing")
                continue
            if "bytes" in info and path.stat().st_size != int(info["bytes"]):
                problems.append(
                    f"{name} has {path.stat().st_size} bytes, not {info['bytes']}"
                )
            elif deep and "sha256" in info and _sha256(path) != info["sha256"]:
                problems.append(f"{name} does not match its sha256")
        for folder in REQUIRED_DIRS:
            for path in sorted((self.path / folder).rglob("*")):
                relative = path.relative_to(self.path).as_posix()
                if (
                    path.is_file()
                    and not path.name.startswith(".")
                    and relative not in listed
                ):
                    problems.append(f"{relative} is not listed in release.json")
        return problems


def open_release(
    release: str | os.PathLike | Release | None = None,
    *,
    offline: bool | None = None,
    pull: bool = True,
) -> Release:
    """Open a release, downloading it first when it is not on disk.

    release: a directory holding `release.json`, a tag (`"v1"`; looked up at
        `$TOKAMAKSI_HOME/releases/<tag>`), or None for `$TOKAMAKSI_RELEASE`, else
        the release this version was cut against.
    offline: never touch the network (default: `TOKAMAKSI_OFFLINE` or
        `HF_HUB_OFFLINE`). A tag that is not on disk then raises `OfflineError`
        naming the command to run.
    pull: when False a missing tag raises instead of being downloaded.

    Raises `AccessError` (private dataset, no token), `OfflineError`,
    `ReleaseError` (a file does not match its checksum, or a folder is not a
    release) and `SchemaError` (a release newer than this package). Opened
    releases are kept per process by resolved path.
    """
    if isinstance(release, Release):
        return release
    value = str(config.default_release() if release is None else release)
    directory = Path(value).expanduser()
    if directory.is_dir() and (directory / "release.json").is_file():
        return _remember(directory)
    if directory.is_dir() and not config.is_tag(value):
        raise ReleaseError(f"{directory} is not a release: it has no release.json")
    if not config.is_tag(value):
        raise ReleaseError(
            f"{value!r} is neither a release directory nor a tag like 'v1'"
        )
    from . import hf

    target = hf.release_dir(value)
    if (target / "release.json").is_file():
        return _remember(target)
    if offline is None:
        offline = config.offline()
    if offline or not pull:
        why = "the network is off" if offline else "pull=False"
        raise OfflineError(
            f"release {value} is not under {hf.releases_dir()} and {why}. "
            f"Run `tokamaksi pull {value}` once with network access (or set "
            f"TOKAMAKSI_HOME to a folder that has it)."
        )
    return _remember(hf.pull(value))


def _remember(path: Path) -> Release:
    key = Path(path).resolve()
    if key not in _OPEN:
        _OPEN[key] = Release(key)
    return _OPEN[key]
