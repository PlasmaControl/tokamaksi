"""One shot's signals, read from a processed-corpus HDF5 file (needs `tokamaksi[h5]`).

A corpus file is `<corpus>/<shot>_processed.h5`. Each group is a diagnostic with a
`xdata` dataset (seconds, float32, `(T,)`) and a `ydata` dataset (`(C, T)` float32,
one row per channel). A group whose `ydata` is `(C, 1)` is a sentinel: the
diagnostic did not run on that shot. Times here are milliseconds from the start of
the shot, the unit the labels use. Live fetching from DIII-D is a later milestone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .errors import MissingPartError, NoDataError

#: A group whose `ydata` has this many samples or fewer is the absent-signal sentinel.
SENTINEL_WIDTH = 1
#: A bare group name draws every channel only up to this many; pick channels past it.
MAX_BARE_CHANNELS = 8


@dataclass(frozen=True)
class Trace:
    """One group's channels over a time window."""

    shot: int
    group: str
    t_ms: np.ndarray  # (T,) float64
    y: np.ndarray  # (C, T) float32
    channels: tuple[int, ...]  # the corpus row of each y row
    decimated: bool = False  # min/max pairs (2 points per bin), not every sample
    source: str = "corpus"

    @property
    def label(self) -> str:
        """`group` or `group[channels]`, as the plot names it."""
        return f"{self.group}[{','.join(str(c) for c in self.channels)}]"


def _import_h5py():
    try:
        import h5py
    except ImportError as error:
        raise MissingPartError(
            "reading shot signals needs h5py: pip install 'tokamaksi[h5]'"
        ) from error
    return h5py


def corpus_dir(corpus: str | os.PathLike | None = None) -> Path:
    """The signal folder: the argument, else `$TOKAMAKSI_CORPUS`; there is no default."""
    chosen = Path(corpus).expanduser() if corpus is not None else config.corpus()
    if chosen is None:
        raise MissingPartError(
            "no signal folder: pass corpus=... or set TOKAMAKSI_CORPUS to the folder "
            "of <shot>_processed.h5 files"
        )
    return chosen


def corpus_path(shot: int, corpus: str | os.PathLike | None = None) -> Path:
    """The file of one shot."""
    return corpus_dir(corpus) / f"{int(shot)}_processed.h5"


def parse(spec: str | tuple) -> tuple[str, tuple[int, ...] | None]:
    """A signal spec as `(group, channels)`; channels None means every channel.

    `"filterscopes"` (every channel), `"filterscopes:3"`, `"filterscopes:0,2,5"`,
    `("filterscopes", 3)` and `("filterscopes", [0, 2])` are all accepted.
    """
    if isinstance(spec, str):
        group, _, rest = spec.partition(":")
        if not rest:
            return group.strip(), None
        try:
            return group.strip(), tuple(int(part) for part in rest.split(","))
        except ValueError:
            raise ValueError(
                f"bad signal spec {spec!r}: expected group or group:i,j"
            ) from None
    group, channels = spec
    if isinstance(channels, (int, np.integer)):
        channels = [channels]
    return str(group), None if channels is None else tuple(int(c) for c in channels)


def available(
    shot: int, corpus: str | os.PathLike | None = None
) -> dict[str, tuple[int, int]]:
    """Every group of the shot that carries data, as `{group: (channels, samples)}`.

    Sentinel groups (the diagnostic did not run) are left out.
    """
    h5py = _import_h5py()
    path = corpus_path(shot, corpus)
    _require_file(path, shot)
    found = {}
    with h5py.File(path, "r") as handle:
        for name, group in handle.items():
            if isinstance(group, h5py.Group) and "ydata" in group:
                rows, width = group["ydata"].shape[0], group["ydata"].shape[-1]
                if width > SENTINEL_WIDTH:
                    found[name] = (int(rows), int(width))
    return found


def _require_file(path: Path, shot: int) -> None:
    if not path.is_file():
        raise NoDataError(
            f"shot {int(shot)} has no signal file {path.name} in {path.parent}: "
            f"the corpus does not cover every shot"
        )


def _bisect(x, value: float, *, right: bool) -> int:
    """`np.searchsorted` on an HDF5 dataset, reading about log2(T) samples."""
    low, high = 0, x.shape[0]
    while low < high:
        mid = (low + high) // 2
        sample = float(x[mid]) * 1000.0
        if sample < value or (right and sample == value):
            low = mid + 1
        else:
            high = mid
    return low


def envelope(
    t_ms: np.ndarray, y: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink `(T,)` times and `(C, T)` values to `2 * bins` points, a minimum and
    a maximum per bin, so spikes survive the thinning. Returns them unchanged when
    there are no more samples than `2 * bins`."""
    n = t_ms.shape[0]
    if n <= 2 * bins:
        return t_ms, y
    edges = np.linspace(0, n, bins + 1).astype(np.int64)[:-1]
    last = np.append(edges[1:], n) - 1
    centre = 0.5 * (t_ms[edges] + t_ms[last])
    low = np.fmin.reduceat(y, edges, axis=1)  # fmin/fmax skip NaN, as plots do
    high = np.fmax.reduceat(y, edges, axis=1)
    out = np.empty((y.shape[0], 2 * bins), dtype=y.dtype)
    out[:, 0::2], out[:, 1::2] = low, high
    return np.repeat(centre, 2), out


def read(
    shot: int,
    signal: str | tuple,
    *,
    t_range: tuple[float, float] | None = None,
    corpus: str | os.PathLike | None = None,
    max_points: int | None = None,
    source: str = "corpus",
) -> Trace:
    """One group of one shot as a `Trace`.

    signal: a spec as in `parse`; a bare group name takes every channel only up to
        `MAX_BARE_CHANNELS`, otherwise name channels (`"mirnov:0,1"`). The corpus
        stores a channel that recorded nothing as NaN: a bare group leaves such
        channels out (`Trace.channels` says which are left), and a named channel
        that is all NaN raises `NoDataError`.
    t_range: `(start, stop)` in ms, applied as an HDF5 slice (a long ECE record is
        never read whole).
    max_points: when the window holds more than twice this many samples, return
        `max_points` minimum/maximum pairs per channel (`Trace.decimated`).
    source: `"corpus"`; `"fdp"` (live DIII-D fetching) is a later milestone.
    """
    if source == "fdp":
        _fdp_stub()
    if source != "corpus":
        raise ValueError(f"source must be 'corpus' or 'fdp', not {source!r}")
    group, wanted = parse(signal)
    h5py = _import_h5py()
    path = corpus_path(shot, corpus)
    _require_file(path, shot)
    with h5py.File(path, "r") as handle:
        if group not in handle:
            raise NoDataError(
                f"shot {int(shot)} has no group {group!r}; it has {', '.join(sorted(handle))}"
            )
        x, y = handle[group]["xdata"], handle[group]["ydata"]
        if y.shape[-1] <= SENTINEL_WIDTH:
            raise NoDataError(
                f"shot {int(shot)} carries the absent-signal sentinel for {group!r} "
                f"(ydata is {y.shape}): that diagnostic did not run"
            )
        rows = tuple(range(y.shape[0])) if wanted is None else wanted
        if wanted is None and len(rows) > MAX_BARE_CHANNELS:
            raise ValueError(
                f"{group!r} has {len(rows)} channels: name some, e.g. '{group}:0,1'"
            )
        bad = [row for row in rows if not 0 <= row < y.shape[0]]
        if bad:
            raise ValueError(f"{group!r} has {y.shape[0]} channels, not {bad}")
        start, stop = 0, x.shape[0]
        if t_range is not None:
            start = _bisect(x, float(t_range[0]), right=False)
            stop = _bisect(x, float(t_range[1]), right=True)
            if stop <= start:
                raise NoDataError(
                    f"shot {int(shot)} {group!r} has no samples in {t_range[0]}-{t_range[1]} ms"
                )
        t_ms = np.asarray(x[start:stop], dtype="float64") * 1000.0
        values = np.stack([y[row, start:stop] for row in rows]).astype("float32")
    # The corpus marks a channel that recorded nothing with NaN throughout.
    dead = np.isnan(values).all(axis=1)
    if dead.any():
        if wanted is not None or dead.all():
            names = [row for row, gone in zip(rows, dead) if gone]
            raise NoDataError(
                f"shot {int(shot)} {group!r} channel(s) {names} hold only NaN in this "
                f"window: nothing was recorded there"
            )
        rows = tuple(row for row, gone in zip(rows, dead) if not gone)
        values = values[~dead]
    decimated = False
    if max_points is not None and t_ms.shape[0] > 2 * max_points:
        t_ms, values = envelope(t_ms, values, max_points)
        decimated = True
    return Trace(int(shot), group, t_ms, values, rows, decimated, "corpus")


def _fdp_stub() -> None:
    """Live fetching is milestone 2; say what it will need, and stop."""
    try:
        import toksearch  # noqa: F401
    except ImportError as error:
        raise MissingPartError(
            "live fetching needs fdp and toksearch from the conda channel ga-fdp and "
            "DIII-D access; it is not part of this release of tokamaksi"
        ) from error
    raise NotImplementedError(
        "fetching from fdp is planned for tokamaksi milestone 2; read the processed "
        "corpus with source='corpus' for now"
    )
