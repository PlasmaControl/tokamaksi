"""Draw a shot's labels, and a few of its signals above them. Needs `tokamaksi[plot]`.

Two rules the drawing never bends. Time nobody assessed is hatched grey: not white,
and not the colour of category 0, which means "assessed, not seen". That includes
rows whose category says so (`unassessed`, `unmarked (not assessed)`) and rows
marked `not_observable`: the 10 ms grid makes all of them -1, so they are not drawn
and the hatch shows through. A span (a stretch of activity) is a pale box with a
coloured edge and an individual event is a dark-edged box with a triangle above its
row, so the two differ by shape and not only colour.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import signals as signals_module
from .schema import GRID_MS
from .taxonomy import Taxonomy

#: Okabe-Ito colours, keyed by category name; any other name takes the next of
#: `FALLBACK` in the order it is first drawn.
CATEGORY_COLOURS = {
    "present": "#0072B2",
    "uncertain": "#E69F00",
    "high": "#0072B2",
    "low": "#D55E00",
    "qh": "#009E73",
    "wpqh": "#CC79A7",
    "imode": "#56B4E9",
    "attached": "#0072B2",
    "detached": "#D55E00",
    "marfe": "#F0E442",
    # Neutral, not blue: a span of "present" is a pale blue and must not be mistaken
    # for assessed-and-absent time.
    "absent": "#E3DCCB",
}
FALLBACK = ("#CC79A7", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#E69F00")
HATCH_EDGE = "#8C8C8C"
#: Category names that mean nobody could or did assess the time (the grid makes them
#: -1); a name containing "not assessed" counts too.
NOT_ASSESSED = ("not_observable", "unassessed")
#: How much of a category's colour a span's pale fill keeps (the rest is white).
SPAN_TINT = 0.6
INK = "#1A1A1A"
TIER_TAG = {"reviewed": "rev", "original": "orig", "derived": "der", "source": "src"}
#: Signal points drawn per channel: a longer window is thinned to min/max pairs.
SIGNAL_POINTS = 3000


def _mpl():
    try:
        import matplotlib  # noqa: F401
        from matplotlib import patches
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise ImportError(
            "plotting needs matplotlib: pip install 'tokamaksi[plot]'"
        ) from error
    return Figure, patches, Line2D


@dataclass
class _Row:
    event: str
    tier: str
    label: str
    frame: pd.DataFrame


def _not_assessed(name: str) -> bool:
    """Whether a category name says this time was not assessed (or not observable)."""
    low = name.strip().lower()
    return low in NOT_ASSESSED or "not assessed" in low


def _order_key(event: str, taxonomy: Taxonomy | None):
    """Parents first, each followed by its subcategories."""
    if taxonomy is None or event not in taxonomy:
        return (10**6, 0, event)
    names = taxonomy.events
    parent = taxonomy.parent(event)
    root = parent if parent in taxonomy else event
    return (names.index(root), 0 if root == event else 1, names.index(event))


def _row_label(
    event: str, tier: str, taxonomy: Taxonomy | None, dataset: str = ""
) -> str:
    name = (
        taxonomy.display(event) if taxonomy is not None and event in taxonomy else event
    )
    indent = (
        "  "
        if taxonomy is not None and event in taxonomy and taxonomy.parent(event)
        else ""
    )
    return f"{indent}{name} [{TIER_TAG.get(tier, tier)}]" + (
        f" {dataset}" if dataset else ""
    )


def _dataset_tags(frame: pd.DataFrame) -> dict[str, str]:
    """A short name for each dataset in `frame` that tells them apart: method and
    version, else the last part of the dataset id, else the whole id."""
    info = frame.drop_duplicates("dataset")

    def last(row) -> str:
        return str(row.dataset).rstrip("/").rsplit("/", 1)[-1]

    def method(row) -> str:
        parts = (str(row.method or ""), str(row.version or ""))
        return " ".join(part for part in parts if part) or last(row)

    tags: dict[str, str] = {}
    for name in (method, last, lambda row: str(row.dataset)):
        tags = {str(row.dataset): name(row) for row in info.itertuples()}
        if len(set(tags.values())) == len(tags):
            break
    return tags


def _make_rows(
    labels: pd.DataFrame,
    taxonomy: Taxonomy | None,
    rows: Sequence[tuple[str, str]] | None,
) -> list[_Row]:
    if rows is None:
        rows = sorted(
            {(e, t) for e, t in zip(labels["event"], labels["tier"])},
            key=lambda pair: (_order_key(pair[0], taxonomy), pair[1]),
        )
    out = []
    for event, tier in rows:
        frame = labels[(labels["event"] == event) & (labels["tier"] == tier)]
        if frame["dataset"].nunique() <= 1:
            out.append(_Row(event, tier, _row_label(event, tier, taxonomy), frame))
            continue
        # Several datasets in one tier (several machine drafts, say) get a row each:
        # on one row their boxes would cover one another.
        tags = _dataset_tags(frame)
        for dataset in sorted(tags, key=lambda name: (tags[name], name)):
            out.append(
                _Row(
                    event,
                    tier,
                    _row_label(event, tier, taxonomy, tags[dataset]),
                    frame[frame["dataset"] == dataset],
                )
            )
    return out


def _extent(labels: pd.DataFrame) -> tuple[float, float] | None:
    if labels.empty:
        return None
    low, high = float(labels["t_start_ms"].min()), float(labels["t_end_ms"].max())
    pad = 0.02 * (high - low) if high > low else 5 * GRID_MS
    return low - pad, high + pad


def _tint(colour: str, weight: float = SPAN_TINT) -> tuple[float, float, float]:
    """`colour` mixed with white: a pale, opaque version of it."""
    from matplotlib.colors import to_rgb

    return tuple(weight * c + (1 - weight) for c in to_rgb(colour))


class _Colours:
    """Colour per category name: fixed for the known names, a cycle for the rest."""

    def __init__(self):
        self._extra: dict[str, str] = {}

    def __call__(self, name: str) -> str:
        if name in CATEGORY_COLOURS:
            return CATEGORY_COLOURS[name]
        if name not in self._extra:
            self._extra[name] = FALLBACK[len(self._extra) % len(FALLBACK)]
        return self._extra[name]


def _draw_rows(ax, rows: list[_Row], x0: float, x1: float):
    """Paint the rows on `ax`; returns the legend handles of what was drawn."""
    _, patches, Line2D = _mpl()
    colour = _Colours()
    minimum = (x1 - x0) / 1500.0  # one pixel on a typical figure
    drawn: dict[str, str] = {}
    skipped: set[str] = set()
    kinds: set[str] = set()
    marks_x, marks_y = [], []
    count = len(rows)
    for index, row in enumerate(rows):
        base = count - 1 - index
        background = patches.Rectangle(
            (x0, base + 0.1),
            x1 - x0,
            0.8,
            facecolor="white",
            edgecolor=HATCH_EDGE,
            hatch="////",
            linewidth=0,
            zorder=1,
        )
        background.set_gid("unassessed")
        ax.add_patch(background)
        if row.frame.empty:
            ax.text(
                0.5 * (x0 + x1),
                base + 0.5,
                f"no {row.tier} labels",
                ha="center",
                va="center",
                fontsize=8,
                color=INK,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1.5,
                },
                zorder=3,
            )
            continue
        for item in row.frame.itertuples():
            name = str(item.category_name)
            if _not_assessed(name):
                # Not drawn: the hatch of unassessed time stays visible here.
                skipped.add(name.strip().lower())
                continue
            face = colour(name)
            drawn[name] = face
            start, stop = float(item.t_start_ms), float(item.t_end_ms)
            width = max(stop - start, minimum)
            kind = item.kind
            if name != "absent":
                kinds.add(kind)
            if kind == "span":
                # A pale tint, not a transparent fill: the hatch of unassessed time
                # must not show through a box that says somebody assessed this.
                style = {"facecolor": _tint(face), "edgecolor": face, "linewidth": 0.8}
            elif kind == "individual":
                style = {"facecolor": face, "edgecolor": INK, "linewidth": 0.9}
                marks_x.append(start + 0.5 * width)
                marks_y.append(base + 0.97)
            else:
                style = {"facecolor": face, "edgecolor": "none", "linewidth": 0}
            box = patches.Rectangle((start, base + 0.1), width, 0.8, zorder=2, **style)
            box.set_gid(kind)
            ax.add_patch(box)
    if marks_x:
        ax.scatter(
            marks_x, marks_y, marker="v", s=14, color=INK, linewidths=0, zorder=4
        )
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.05, count + 0.05)
    ax.set_yticks([count - 1 - i + 0.5 for i in range(count)])
    ax.set_yticklabels([row.label for row in rows], fontsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("time (ms)")

    hatch_name = (
        "unassessed or not observable" if "not_observable" in skipped else "unassessed"
    )
    handles = [
        patches.Patch(
            facecolor="white", edgecolor=HATCH_EDGE, hatch="////", label=hatch_name
        )
    ]
    handles += [
        patches.Patch(facecolor=face, label=name) for name, face in drawn.items()
    ]
    if "span" in kinds:
        handles.append(
            patches.Patch(facecolor=_tint("#7F7F7F"), edgecolor="#7F7F7F", label="span")
        )
    if "individual" in kinds:
        handles.append(
            Line2D(
                [],
                [],
                marker="v",
                color=INK,
                linestyle="none",
                markersize=5,
                label="individual",
            )
        )
    return handles


def timeline(
    labels: pd.DataFrame,
    *,
    taxonomy: Taxonomy | None = None,
    ax=None,
    t_range: tuple[float, float] | None = None,
    rows: Sequence[tuple[str, str]] | None = None,
    legend: bool = True,
):
    """Draw already-filtered labels of one shot as stacked rows, one per (event, tier).

    rows: the `(event, tier)` pairs to draw, in order; a pair with no labels still
    gets its row, saying so. Default: every pair present, parents before their
    subcategories. A pair whose labels come from several datasets gets one row per
    dataset, named after its method and version. Rows whose category says not
    assessed or not observable are left as hatch. `taxonomy` supplies display names;
    without it event ids are shown. Returns the Axes; nothing is shown or saved.
    """
    Figure, _, _ = _mpl()
    if labels["shot"].nunique() > 1:
        raise ValueError("timeline draws one shot; filter the labels to one shot first")
    row_list = _make_rows(labels, taxonomy, rows)
    window = t_range or _extent(labels) or (0.0, 1000.0)
    if ax is None:
        figure = Figure(
            figsize=(10, max(1.6, 0.42 * len(row_list) + 1.0)), layout="constrained"
        )
        ax = figure.subplots()
    handles = _draw_rows(ax, row_list, float(window[0]), float(window[1]))
    if legend:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            frameon=False,
        )
    return ax


def _signal_panel(ax, trace) -> None:
    many = trace.y.shape[0] > 1
    for row, channel in zip(trace.y, trace.channels):
        ax.plot(trace.t_ms, row, linewidth=0.7, label=f"ch {channel}" if many else None)
    ax.set_ylabel(trace.label, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    if many:
        ax.legend(
            fontsize=6, frameon=False, ncol=min(trace.y.shape[0], 4), loc="upper right"
        )


def shot(
    shot: int,
    events: str | Sequence[str] | None = None,
    *,
    signals: str | Sequence[str] | None = None,
    tier: str | Sequence[str] = "reviewed",
    dataset: str | Sequence[str] | None = None,
    t_range: tuple[float, float] | None = None,
    release: str | os.PathLike | None = None,
    corpus: str | os.PathLike | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """One shot as signal panels over its label timeline; returns a matplotlib Figure.

    events: event ids to draw; None draws every event that has rows for this shot.
        A requested (event, tier) with no rows still gets a row that says so.
    signals: signal specs to stack above the labels, read from the processed corpus
        (`tokamaksi.signals.parse`: `"neutron_rate:0"`, `"bolo:0,1"`; `corpus=` or
        `$TOKAMAKSI_CORPUS` names the folder; needs `tokamaksi[h5]`).
    t_range: `(start, stop)` in ms; default the labelled extent plus 2 %.
    Nothing falls back to another tier. The figure is never shown or saved here.
    """
    from .labels import load_labels
    from .release import open_release

    Figure, _, _ = _mpl()
    rel = open_release(release)
    taxonomy = rel.taxonomy()
    tiers = [tier] if isinstance(tier, str) else list(tier)
    labels = load_labels(
        events, tier=tiers, shots=int(shot), dataset=dataset, release=rel
    )
    pairs = None
    if events is not None:
        names = [events] if isinstance(events, str) else list(events)
        pairs = sorted(
            ((e, t) for e in names for t in tiers),
            key=lambda pair: (_order_key(pair[0], taxonomy), tiers.index(pair[1])),
        )
    row_list = _make_rows(labels, taxonomy, pairs)
    if not row_list:
        row_list = [
            _Row("", t, f"(no events) [{TIER_TAG.get(t, t)}]", labels) for t in tiers
        ]
    window = t_range or _extent(labels)
    specs = (
        []
        if signals is None
        else ([signals] if isinstance(signals, (str, tuple)) else list(signals))
    )
    traces = [
        signals_module.read(
            shot, spec, t_range=window, corpus=corpus, max_points=SIGNAL_POINTS
        )
        for spec in specs
    ]
    if window is None and traces:
        window = (min(t.t_ms.min() for t in traces), max(t.t_ms.max() for t in traces))
    window = window or (0.0, 1000.0)

    label_height = max(1.1, 0.4 * len(row_list) + 0.7)
    height = label_height + 1.7 * len(traces)
    figure = Figure(figsize=figsize or (10.5, height), layout="constrained")
    axes = np.atleast_1d(
        figure.subplots(
            len(traces) + 1,
            1,
            sharex=True,
            height_ratios=[1.7] * len(traces) + [label_height],
        )
    )
    for ax, trace in zip(axes[:-1], traces):
        _signal_panel(ax, trace)
    handles = _draw_rows(axes[-1], row_list, float(window[0]), float(window[1]))
    figure.legend(handles=handles, loc="outside right upper", fontsize=7, frameon=False)
    figure.suptitle(title or f"shot {int(shot)}  (release {rel.tag})", fontsize=10)
    return figure
