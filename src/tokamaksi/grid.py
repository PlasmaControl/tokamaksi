"""The 10 ms grid: runs of equal frames per shot and event, decoded to arrays."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .errors import MissingPartError
from .release import Release, open_release
from .schema import GRID_COLUMNS, GRID_MS, UNASSESSED, require_columns


@dataclass(frozen=True)
class ShotGrid:
    """One shot on the 10 ms grid: `frames[i, k]` is the category of `events[i]` in
    the frame `[t0_ms + 10 k, t0_ms + 10 (k + 1))`, or `-1` where nobody assessed it."""

    shot: int
    events: tuple[str, ...]
    t0_ms: int
    frames: np.ndarray  # (n_events, n_frames) int8

    @property
    def time_ms(self) -> np.ndarray:
        """The start time of every frame."""
        return self.t0_ms + GRID_MS * np.arange(self.frames.shape[1])

    def categories(self, event: str) -> np.ndarray:
        """The category per frame of one event."""
        return self.frames[self.events.index(event)]

    def assessed(self, event: str | None = None) -> np.ndarray:
        """Frames somebody assessed (not -1): of one event, or of any event."""
        if event is not None:
            return self.categories(event) != UNASSESSED
        return (self.frames != UNASSESSED).any(axis=0)

    def at(self, t_ms: float) -> dict[str, int]:
        """The category of every event in the frame holding `t_ms` (-1 outside)."""
        k = int(np.floor((t_ms - self.t0_ms) / GRID_MS))
        inside = 0 <= k < self.frames.shape[1]
        return {
            name: int(self.frames[i, k]) if inside else UNASSESSED
            for i, name in enumerate(self.events)
        }


class Grid:
    """The runs of one grid file (`gold`, `silver` or `holdout`) of a release."""

    def __init__(self, name: str, runs: pd.DataFrame):
        self.name = name
        self.runs = runs
        self.shots: list[int] = sorted(int(s) for s in runs["shot"].unique())

    def decode(self, shot: int, events: Sequence[str] | None = None) -> ShotGrid:
        """The frames of one shot, `-1` where no run covers them.

        events: keep only these (in this order; an event with no run on this shot
        is all -1). The time axis is the same whichever events you keep.
        """
        shot_runs = self.runs[self.runs["shot"] == int(shot)]
        if shot_runs.empty:
            raise MissingPartError(f"shot {int(shot)} is not in the {self.name} grid")
        names, t0_ms, frames = decode_grid(shot_runs, shot, events)
        return ShotGrid(int(shot), tuple(names), t0_ms, frames)


def decode_grid(
    runs: pd.DataFrame, shot: int, events: Sequence[str] | None = None
) -> tuple[list[str], int, np.ndarray]:
    """One shot's (events, t0 in ms, event x frame categories) back from its runs.

    Ported from FusionAIHub `labeler.events.hf_export.decode_grid` at commit
    52c2a2bd; `events` is new. The time span comes from all of the shot's runs, so
    a subset of events lines up with the full decode.
    """
    runs = runs[runs.shot == int(shot)]
    first = int(runs.frame.min())
    n = int((runs.frame + runs.n_frames).max()) - first
    names = list(dict.fromkeys(runs.event)) if events is None else list(events)
    out = np.full((len(names), n), UNASSESSED, dtype=np.int8)
    for row in runs.itertuples():
        if row.event not in names:
            continue
        start = int(row.frame) - first
        out[names.index(row.event), start : start + int(row.n_frames)] = row.category
    return names, first * GRID_MS, out


def load_grid(
    which: str = "gold", *, release: str | os.PathLike | Release | None = None
) -> Grid:
    """The 10 ms grid runs of `which`: `"gold"` (split `test`, evaluation only) or
    `"silver"`. `MissingPartError` names the grids the release has (v0.2.1 ships
    gold and silver only)."""
    rel = open_release(release)
    path = rel.path / "grid" / f"{which}.parquet"
    if not path.is_file():
        have = sorted(p.stem for p in (rel.path / "grid").glob("*.parquet"))
        raise MissingPartError(
            f"release {rel.tag} has no {which} grid; it has {', '.join(have) or 'none'}"
        )
    runs = pq.read_table(path).to_pandas()
    require_columns(runs.columns, GRID_COLUMNS, f"grid/{which}.parquet")
    return Grid(which, runs)
