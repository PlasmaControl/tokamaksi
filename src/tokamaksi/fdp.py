"""The all-shot 1 ms model labels: one database per event, read lazily.

A database is a folder of `spans.parquet`, `shots.parquet`, `valid.csv`, `meta.json`
and `README.md` (docs/schema.md, "All-shot model labels"). They are model output,
nobody reviewed them, and they sit on a 1 ms grid where the release's grid is 10 ms.
A ms inside a labeled shot's assessed window that no span covers is 0 (absent; for
confinement, in no segment). Time outside the window was not assessed and decodes
to -1, as on the 10 ms grid.

Where the databases are: the `root=` argument, else `$TOKAMAKSI_FDP_LABELS`, else
the `fdp_1ms/` folder of a release on disk. Under it, `<event>/` is a database when
it holds `meta.json`, else its one subfolder that does.
"""

from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .errors import MissingPartError, NoDataError, TokamaksiError, UnknownEventError
from .release import Release, open_release
from .schema import FDP_DIR, UNASSESSED, require_columns

#: The frame length of these tables, in ms.
FRAME_MS = 1
#: The `shots.parquet` status of a shot with labels; any other says why it has none.
LABELED = "labeled"
#: The codes of an event whose spans carry no `category`.
BINARY = {0: "absent", 1: "present"}

_SHOT_COLUMNS = ("shot", "status", "window_start_ms", "window_end_ms")
_SPAN_COLUMNS = ("shot", "t_start_ms", "t_end_ms")


def _root(
    root: str | os.PathLike | None = None,
    release: str | os.PathLike | Release | None = None,
) -> Path:
    if root is not None:
        path = Path(root).expanduser()
        if not path.is_dir():
            raise MissingPartError(f"{path} is not a folder")
        return path
    chosen = config.fdp_labels()
    if chosen is not None:
        if not chosen.is_dir():
            raise MissingPartError(
                f"TOKAMAKSI_FDP_LABELS={chosen} is not a folder here (is it mounted on "
                f"this node?)"
            )
        return chosen
    try:
        rel = open_release(release, pull=False)
    except TokamaksiError:
        if release is not None:
            raise
        rel = None
    if rel is not None and (rel.path / FDP_DIR).is_dir():
        return rel.path / FDP_DIR
    where = f"; release {rel.tag} has no {FDP_DIR}/" if rel is not None else ""
    raise MissingPartError(
        "no 1 ms model labels: pass root=... or set TOKAMAKSI_FDP_LABELS to the folder "
        f"holding one <event>/ folder per event{where}"
    )


def _databases(folder: Path) -> list[Path]:
    """`folder` itself when it holds `meta.json`, else its subfolders that do."""
    if (folder / "meta.json").is_file():
        return [folder]
    return sorted(p.parent for p in folder.glob("*/meta.json"))


def events(
    root: str | os.PathLike | None = None,
    *,
    release: str | os.PathLike | Release | None = None,
) -> list[str]:
    """The events with a 1 ms database under the root (see the module docstring)."""
    base = _root(root, release)
    return sorted(p.name for p in base.iterdir() if p.is_dir() and _databases(p))


def open(
    event: str,
    root: str | os.PathLike | None = None,
    *,
    release: str | os.PathLike | Release | None = None,
) -> FdpLabels:
    """One event's 1 ms model labels; the tables are read on first use.

    root: the folder holding one `<event>/` folder per event; default
        `$TOKAMAKSI_FDP_LABELS`, else the `fdp_1ms/` folder of `release` (a tag or a
        directory already on disk; nothing is downloaded).
    Raises `MissingPartError` when no root is found, `UnknownEventError` when the root
    has no folder for `event`, and `NoDataError` when that folder holds no database
    (not published yet) or more than one. The first call in a process reminds, as a
    `TermsWarning`, that tier `derived` is model output.
    """
    base = _root(root, release)
    folder = base / event
    if not folder.is_dir():
        have = ", ".join(events(base)) or "none"
        raise UnknownEventError(
            f"no 1 ms model labels for {event!r} under {base}; the events there: {have}"
        )
    found = _databases(folder)
    if not found:
        raise NoDataError(f"{folder} holds no database (no meta.json): not published?")
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        raise NoDataError(
            f"{folder} holds {len(found)} databases ({names}); keep one, or pass "
            f"root= a folder that has one"
        )
    from .labels import _remind

    _remind("derived")
    return FdpLabels(event, found[0])


class FdpLabels:
    """One event's all-shot 1 ms model labels. Nothing is written.

    `shots` has one row per plasma candidate (`status` is `labeled` or why there are
    no labels; `window_start_ms`, `window_end_ms` bound the assessed window,
    half-open), `spans` the present stretches `[t_start_ms, t_end_ms)` (confinement
    adds `category`), `valid` every shot number of the range with `valid` 1 when it
    has labels, and `meta` the producer's notes as a dict (its keys differ between
    events: read `meta["caveats"]`). AE's `notch` column is for display only: never
    filter on it.
    """

    def __init__(self, event: str, path: str | os.PathLike):
        self.event = event
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"FdpLabels({self.event!r}, {str(self.path)!r})"

    @cached_property
    def meta(self) -> dict:
        return json.loads((self.path / "meta.json").read_text(encoding="utf-8"))

    @cached_property
    def shots(self) -> pd.DataFrame:
        frame = pd.read_parquet(self.path / "shots.parquet")
        require_columns(frame.columns, _SHOT_COLUMNS, f"{self.event} shots.parquet")
        return frame

    @cached_property
    def spans(self) -> pd.DataFrame:
        frame = pd.read_parquet(self.path / "spans.parquet")
        require_columns(frame.columns, _SPAN_COLUMNS, f"{self.event} spans.parquet")
        return frame

    @cached_property
    def valid(self) -> pd.DataFrame:
        return pd.read_csv(self.path / "valid.csv")

    @cached_property
    def labeled(self) -> np.ndarray:
        """The shots with labels, ascending."""
        shots = self.shots
        return np.sort(shots.loc[shots["status"] == LABELED, "shot"].to_numpy("int64"))

    @cached_property
    def categories(self) -> dict[int, str]:
        """What each code means: `meta.json`'s `categories` when it has them, else
        0 absent and 1 present."""
        named = self.meta.get("categories")
        if isinstance(named, dict) and named:
            return {int(code): str(text) for code, text in named.items()}
        return dict(BINARY)

    @cached_property
    def _rows(self) -> pd.DataFrame:
        return self.shots.set_index("shot")

    @cached_property
    def _index(self) -> tuple[np.ndarray, ...]:
        spans = self.spans
        order = np.argsort(spans["shot"].to_numpy("int64"), kind="stable")
        values = (
            spans["category"].to_numpy("int64")
            if "category" in spans
            else np.ones(len(spans), dtype="int64")
        )
        return (
            spans["shot"].to_numpy("int64")[order],
            spans["t_start_ms"].to_numpy("int64")[order],
            spans["t_end_ms"].to_numpy("int64")[order],
            values[order],
        )

    def status(self, shot: int) -> str:
        """`labeled`, or why the shot has no labels (`meta["statuses"]` explains)."""
        if int(shot) not in self._rows.index:
            raise NoDataError(
                f"shot {int(shot)} is not a plasma candidate of the {self.event} 1 ms "
                f"labels"
            )
        return str(self._rows.at[int(shot), "status"])

    def window(self, shot: int) -> tuple[int, int]:
        """The assessed window `[start, end)` in ms of a labeled shot."""
        status = self.status(shot)
        if status != LABELED:
            why = self.meta.get("statuses", {}).get(status)
            raise NoDataError(
                f"shot {int(shot)} has no {self.event} 1 ms labels: {status}"
                + (f" ({why})" if why else "")
            )
        row = self._rows.loc[int(shot)]
        return int(row["window_start_ms"]), int(row["window_end_ms"])

    def _spans_of(self, shot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shots, starts, ends, values = self._index
        lo, hi = np.searchsorted(shots, [int(shot), int(shot) + 1])
        return starts[lo:hi], ends[lo:hi], values[lo:hi]

    def decode(
        self, shot: int, t_range: tuple[float, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """`(t_ms, codes)` on the 1 ms grid: `codes[k]` holds for `[t_ms[k],
        t_ms[k] + 1)`.

        codes (int8): 1 present (confinement: the category code), 0 absent (in no
        segment), -1 outside the assessed window.
        t_range: `(start, stop)` in ms; default the assessed window, so no -1.
        Raises `NoDataError` for a shot without labels, naming its status.
        """
        start, end = self.window(shot)
        lo, hi = (
            (start, end)
            if t_range is None
            else (int(np.floor(t_range[0])), int(np.ceil(t_range[1])))
        )
        if hi < lo:
            raise ValueError(f"t_range {t_range} runs backwards")
        codes = np.full(hi - lo, UNASSESSED, dtype=np.int8)
        a, b = max(start, lo), min(end, hi)
        if b > a:
            codes[a - lo : b - lo] = 0
        for s, e, value in zip(*self._spans_of(shot)):
            s, e = max(int(s), a), min(int(e), b)
            if e > s:
                codes[s - lo : e - lo] = value
        return np.arange(lo, hi, dtype=np.int64), codes

    def at(self, shot: int, t_ms: float) -> int:
        """The code at `t_ms`: as `decode`, -1 outside the assessed window."""
        start, end = self.window(shot)
        if not start <= t_ms < end:
            return UNASSESSED
        starts, ends, values = self._spans_of(shot)
        hit = np.flatnonzero((starts <= t_ms) & (t_ms < ends))
        return int(values[hit[-1]]) if hit.size else 0


#: `open` is left out so that `from tokamaksi.fdp import *` keeps the builtin.
__all__ = ["FRAME_MS", "FdpLabels", "LABELED", "events"]
