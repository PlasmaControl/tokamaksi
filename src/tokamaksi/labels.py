"""Interval labels from a release, as plain DataFrames."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .errors import TermsWarning
from .release import Release, open_release
from .schema import LABEL_COLUMNS, SPLITS, TIER_ORDER, require_columns

#: The reminders already shown in this process (each is shown once).
_WARNED: set[str] = set()

_REMINDERS = {
    "original": "tier 'original' is third-party label tables under their owners' terms; "
    "do not redistribute them",
    "derived": "tier 'derived' is rule or model output, not human labels",
    "source": "tier 'source' is machine drafts that nobody reviewed: they are not labels",
    "test": "split 'test' is the gold shots: use them to evaluate, never to train or tune",
}


def _remind(key: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(_REMINDERS[key], TermsWarning, stacklevel=3)


def _as_list(value, name: str, allowed: Sequence[str] | None = None) -> list | None:
    if value is None:
        return None
    items = [value] if isinstance(value, (str, int, np.integer)) else list(value)
    if allowed is not None:
        bad = [item for item in items if item not in allowed]
        if bad:
            raise ValueError(f"unknown {name} {bad}; choose from {list(allowed)}")
    return items


def _kind(iscrowd: pd.Series) -> pd.Series:
    """span (iscrowd True), individual (False) or unspecified (null)."""
    flag = iscrowd.astype("boolean")
    kind = np.where(
        flag.isna(), "unspecified", np.where(flag.fillna(False), "span", "individual")
    )
    return pd.Series(kind, index=iscrowd.index, dtype=object)


def load_labels(
    event: str | Sequence[str] | None = None,
    *,
    tier: str | Sequence[str] = "reviewed",
    split: str | Sequence[str] | None = None,
    shots: int | Sequence[int] | None = None,
    dataset: str | Sequence[str] | None = None,
    release: str | os.PathLike | Release | None = None,
) -> pd.DataFrame:
    """Interval labels as one DataFrame, sorted by shot, event, tier, t_start_ms.

    event: an event id (`"edge_localized_mode"`) or a subcategory id (`"mhd_n1.m2"`),
        or a list; None for all. Unknown ids raise `UnknownEventError` naming the
        closest ids. A parent does not include its children.
    tier: `"reviewed"` (saved by a person on the review page: the labels),
        `"original"` (third-party tables, reformatted), `"derived"` (rule or model
        output) or `"source"` (machine drafts, not labels); a list for several. It
        never falls back to another tier: a shot with no rows in the tier you asked
        for is absent from the result.
    split: `"train"`, `"holdout"` or `"test"` (the gold shots: evaluation only).
        None: every split.
    shots, dataset: filters pushed into the parquet read.

    The columns are the release's (shot, event, category, category_name, t_start_ms,
    t_end_ms, confidence, iscrowd, attrs, tier, layer, dataset, method, version,
    reviewer, split) plus `kind`: `"span"` (iscrowd True: a stretch of activity whose
    events are not counted one by one), `"individual"` (iscrowd False: one countable
    event) or `"unspecified"` (iscrowd null, as on every `absent` row). Intervals
    are half-open `[t_start_ms, t_end_ms)` on a 10 ms grid.

    Missing is not zero: time with no row was not assessed. A row of category 0 says
    "assessed, not seen". `df.attrs["tokamaksi"]` records the release and the query.
    Each reminder about a tier other than `reviewed`, and about the test split, is
    shown once per process as a `TermsWarning`.
    """
    rel = open_release(release)
    taxonomy = rel.taxonomy()
    events = _as_list(event, "event")
    for name in events or ():
        if name not in taxonomy:
            taxonomy.display(name)  # raises UnknownEventError with the neighbours
    tiers = _as_list(tier, "tier", TIER_ORDER)
    splits = _as_list(split, "split", SPLITS)
    shot_list = None if shots is None else [int(s) for s in _as_list(shots, "shots")]
    datasets = _as_list(dataset, "dataset")

    filters = []
    if events is not None:
        filters.append(("event", "in", events))
    if shot_list is not None:
        filters.append(("shot", "in", shot_list))
    if datasets is not None:
        filters.append(("dataset", "in", datasets))

    frames = []
    for name in tiers:
        for part in splits or SPLITS:
            path = rel.path / "labels" / name / f"{part}.parquet"
            if not path.is_file():
                continue
            table = pq.read_table(path, filters=filters or None)
            require_columns(
                table.column_names, LABEL_COLUMNS, f"labels/{name}/{part}.parquet"
            )
            frames.append(
                table.to_pandas(types_mapper={pa.bool_(): pd.BooleanDtype()}.get)
            )
    if frames:
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.DataFrame({name: pd.Series(dtype=object) for name in LABEL_COLUMNS})
    frame["iscrowd"] = frame["iscrowd"].astype("boolean")
    frame["kind"] = _kind(frame["iscrowd"])
    order = {name: i for i, name in enumerate(TIER_ORDER)}
    frame = (
        frame.assign(_tier=frame["tier"].map(order))
        .sort_values(["shot", "event", "_tier", "t_start_ms"], kind="stable")
        .drop(columns="_tier")
        .reset_index(drop=True)
    )
    frame.attrs["tokamaksi"] = {
        "release": rel.tag,
        "commit": rel.commit,
        "query": {
            "event": events,
            "tier": tiers,
            "split": splits,
            "shots": shot_list,
            "dataset": datasets,
        },
    }
    for name in tiers:
        if name != "reviewed":
            _remind(name)
    if (frame["split"] == "test").any():
        _remind("test")
    return frame


def spans(df: pd.DataFrame) -> pd.DataFrame:
    """The rows of `df` with iscrowd True (a stretch of activity)."""
    return df[df["kind"] == "span"]


def individuals(df: pd.DataFrame) -> pd.DataFrame:
    """The rows of `df` with iscrowd False (one countable event)."""
    return df[df["kind"] == "individual"]


def unspecified(df: pd.DataFrame) -> pd.DataFrame:
    """The rows of `df` with iscrowd null, which includes every `absent` row."""
    return df[df["kind"] == "unspecified"]


def at(
    df: pd.DataFrame, shot: int, t_ms: float, event: str | None = None
) -> pd.DataFrame:
    """The rows of `shot` covering `t_ms` (`t_start_ms <= t_ms < t_end_ms`); a
    zero-length row covers its own instant."""
    inside = (df["shot"] == int(shot)) & (df["t_start_ms"] <= t_ms)
    instant = (df["t_start_ms"] == df["t_end_ms"]) & (df["t_start_ms"] == t_ms)
    inside &= (t_ms < df["t_end_ms"]) | instant
    if event is not None:
        inside &= df["event"] == event
    return df[inside]


__all__ = ["load_labels", "spans", "individuals", "unspecified", "at"]
