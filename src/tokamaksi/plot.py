"""Draw a shot's labels, and a few of its signals above them. Needs `tokamaksi[plot]`.

Two rules the drawing never bends. Time nobody assessed is hatched grey: not white,
and not the colour of category 0, which means "assessed, not seen". That includes
rows whose category says so (`unassessed`, `unmarked (not assessed)`), rows marked
`not_observable`, and category 0 of an event whose 0 does not mean "absent"
(`Taxonomy.zero`: confinement and detachment, where 0 only says nothing was marked)
outside the owner's own save: the 10 ms grid makes all of them -1, so they are not
drawn and the hatch shows through. A span (a stretch of activity) is a pale box with
a coloured edge and an individual event is a dark-edged box with a triangle above its
row, so the two differ by shape and not only colour.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import signals as signals_module
from .schema import GRID_MS, TIER_ORDER
from .taxonomy import Taxonomy

#: Okabe-Ito colours, keyed by the taxonomy's name of a category (`high`, not the
#: `high (H)` a producer wrote); a name that is not here takes a colour of `FALLBACK`.
#: `test_real_release` fails when a release names a category that is not here.
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
    # minimum_safety_factor is ordinal, low < hybrid < elevated < high: a ramp from
    # `low` through light green and light blue to `high`. Any two of the ramp,
    # `uncertain` and `absent` are over 15 CIELAB units apart under each of the three
    # colour-vision deficiencies (tests/test_plot.py checks it).
    "hybrid": "#B2DF8A",
    "elevated": "#88CCEE",
    # Neutral, not blue: a span of "present" is a pale blue and must not be mistaken
    # for assessed-and-absent time.
    "absent": "#E3DCCB",
}
#: For names not above, by the position of the category in its event's list in the
#: taxonomy. None of them is in `CATEGORY_COLOURS`, so an unknown name never looks like
#: a known category.
FALLBACK = ("#882255", "#332288", "#117733", "#AA4499", "#999933", "#6E4B2A")
HATCH_EDGE = "#8C8C8C"
#: Category names that mean nobody could or did assess the time (the grid makes them
#: -1); a name containing "not assessed" counts too, and `_` and a space are the same.
NOT_ASSESSED = ("not_observable", "unassessed")
#: How much of a category's colour a span's pale fill keeps (the rest is white).
SPAN_TINT = 0.6
INK = "#1A1A1A"
TIER_TAG = {"reviewed": "rev", "original": "orig", "derived": "der", "source": "src"}
#: The tier tag of a 1 ms model row (`shot(..., model=True)`): model output, not labels.
MODEL_TAG = "1 ms model"
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


def _plain(name: str) -> str:
    """`name` in lower case with `_` and runs of space as a single space."""
    return " ".join(name.replace("_", " ").lower().split())


def _not_assessed(name: str) -> bool:
    """Whether a category name says this time was not assessed (or not observable)."""
    low = _plain(name)
    return low in {_plain(n) for n in NOT_ASSESSED} or "not assessed" in low


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
    tiers: Sequence[str] = TIER_ORDER,
) -> list[_Row]:
    """Rows for `rows`, or for every (event, tier) in `labels`: parents before their
    subcategories, then the tiers in the order of `tiers` (any other after, by name)."""
    if rows is None:
        order = list(tiers)
        rows = sorted(
            {(e, t) for e, t in zip(labels["event"], labels["tier"])},
            key=lambda pair: (
                _order_key(pair[0], taxonomy),
                order.index(pair[1]) if pair[1] in order else len(order),
                pair[1],
            ),
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


_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")


def _category_id(category: object) -> int | None:
    """A row's category as an int, None when it has none."""
    try:
        return int(category)
    except (TypeError, ValueError):
        return None


def _colour_name(
    taxonomy: Taxonomy | None, event: str, category: object, name: str
) -> str:
    """The name a category's colour goes by.

    Producers write names such as `high (H)` or `uncertain (segment below the
    confidence floor)` where the taxonomy says `high` and `uncertain`: this is the
    taxonomy's name for the category id, else the row's own name without a trailing
    parenthesis. (A row is also judged not assessed on its own name; see `_draw_rows`.)
    """
    cid = _category_id(category)
    if taxonomy is not None and event in taxonomy and cid is not None:
        known = taxonomy.categories(event).get(cid)
        if known:
            return known
    return _SUFFIX.sub("", name) or name


def _legend_name(key: str, name: str) -> str:
    """The legend entry of a category: its colour name `key`, or the producer's own
    `name` when that says more (`marfe (candidate)`, `uncertain (segment below the
    confidence floor)`). An abbreviation in parentheses (`high (H)`) adds nothing."""
    raw = " ".join(name.split())
    if raw.lower() == key.lower():
        return key
    hit = _SUFFIX.search(raw)
    if hit and _SUFFIX.sub("", raw).lower() == key.lower():
        inner = hit.group().strip()[1:-1].strip().lower()
        if inner and key.lower().startswith(inner):
            return key
    return raw


def _position(taxonomy: Taxonomy | None, event: str, category: object) -> int | None:
    """Where a category sits in the taxonomy's list for its event, else None."""
    cid = _category_id(category)
    if taxonomy is None or event not in taxonomy or cid is None:
        return None
    ids = list(taxonomy.categories(event))
    return ids.index(cid) if cid in ids else None


def _zero_words(taxonomy: Taxonomy | None, event: str) -> str | None:
    """What category 0 says for `event` when that is not "absent: assessed and not
    seen" (confinement: "the owner saved no regime here"); None where it is absent."""
    if taxonomy is None:
        return None
    words = taxonomy.zero(event)
    return None if words.lower().startswith("absent") else words


def _owners_save(item) -> bool:
    """Whether a label row is the owner's own save (reviewed tier, layer `saved`)."""
    return (
        getattr(item, "tier", None) == "reviewed"
        and getattr(item, "layer", None) == "saved"
    )


class _Colours:
    """Colour per category name: fixed for the known names. Any other takes the
    colour of `FALLBACK` at its `position` (that of its category in the event's list
    in the taxonomy), so within an event it is the same in every figure; without a
    position, the next index in draw order. A colour another name already holds in
    this figure is skipped, so no two names share one."""

    def __init__(self):
        self._extra: dict[str, str] = {}

    def __call__(self, name: str, position: int | None = None) -> str:
        if name in CATEGORY_COLOURS:
            return CATEGORY_COLOURS[name]
        if name not in self._extra:
            index = len(self._extra) if position is None else position
            used = {colour.lower() for colour in self._extra.values()}
            while self._next(index).lower() in used:
                index += 1
            self._extra[name] = self._next(index)
        return self._extra[name]

    @staticmethod
    def _next(index: int) -> str:
        if index < len(FALLBACK):
            return FALLBACK[index]
        from matplotlib.colors import hsv_to_rgb, to_hex

        # Past the list: golden-ratio hues keep spreading the new ones apart.
        hue = (0.618033988749895 * index) % 1.0
        return to_hex(hsv_to_rgb((hue, 0.6, 0.55)))


def _draw_rows(
    ax, rows: list[_Row], x0: float, x1: float, taxonomy: Taxonomy | None = None
):
    """Paint the rows on `ax`; returns the legend handles of what was drawn.

    A category is coloured by its taxonomy name (see `_colour_name`), so `high` and
    `high (H)` are one colour; the legend lists it by that name unless the producer's
    own name says more (`_legend_name`). A row is left to the hatch when its category
    name says not assessed (on its own name or the taxonomy's), and when it is category
    0 of an event whose 0 is not "absent" (`Taxonomy.zero`) and not the owner's save;
    the owner's own 0 is painted, and listed by the taxonomy's words.
    """
    _, patches, Line2D = _mpl()
    colour = _Colours()
    minimum = (x1 - x0) / 1500.0  # one pixel on a typical figure
    drawn: dict[str, str] = {}
    not_observable = False
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
            key = _colour_name(taxonomy, item.event, item.category, name)
            if _not_assessed(name) or _not_assessed(key):
                # Not drawn: the hatch of unassessed time stays visible here.
                not_observable |= "observable" in _plain(name + " " + key)
                continue
            words = None
            if _category_id(item.category) == 0:
                words = _zero_words(taxonomy, item.event)
                if words and not _owners_save(item):
                    continue  # nothing was marked here, not even "absent"
            face = colour(key, _position(taxonomy, item.event, item.category))
            drawn[words or _legend_name(key, name)] = face
            start, stop = float(item.t_start_ms), float(item.t_end_ms)
            width = max(stop - start, minimum)
            kind = item.kind
            if key != "absent":
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

    hatch_name = "unassessed or not observable" if not_observable else "unassessed"
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
    subcategories, each event's tiers in the order of `schema.TIER_ORDER`. A pair
    whose labels come from several datasets gets one row per dataset, named after
    its method and version. Rows whose category says not assessed or not observable,
    and category 0 of confinement or detachment outside the owner's save (the taxonomy's
    `zero` says it is not "absent"), are left as hatch. `taxonomy` supplies display
    names, the category names that colours go by and what category 0 means; without it
    event ids are shown, a category goes by its own name less a trailing parenthesis
    and every category 0 is painted. Returns the Axes; nothing is shown or saved.
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
    handles = _draw_rows(ax, row_list, float(window[0]), float(window[1]), taxonomy)
    if legend:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            frameon=False,
        )
    return ax


def _model_runs(shot: int, db, taxonomy: Taxonomy | None) -> pd.DataFrame:
    """One shot of a 1 ms database as label-like rows, one per run of equal codes."""
    from .fdp import BINARY

    t_ms, codes = db.decode(shot)
    if codes.size == 0:
        return pd.DataFrame(columns=["t_start_ms", "t_end_ms"])
    cut = np.flatnonzero(np.diff(codes)) + 1
    starts, stops = np.r_[0, cut], np.r_[cut, codes.size]
    names = dict(BINARY if "category" not in db.spans else db.categories)
    if taxonomy is not None and db.event in taxonomy:
        names.update(taxonomy.categories(db.event))
    category = codes[starts].astype(int)
    return pd.DataFrame(
        {
            "shot": int(shot),
            "event": db.event,
            "category": category,
            "category_name": [names.get(c, str(c)) for c in category],
            "t_start_ms": t_ms[starts].astype(float),
            "t_end_ms": (t_ms[stops - 1] + 1).astype(float),
            "tier": "derived",
            "layer": "model",
            "kind": "unspecified",
        }
    )


def _model_rows(
    shot: int,
    events: Sequence[str] | None,
    taxonomy: Taxonomy | None,
    release,
) -> list[_Row]:
    """A 1 ms model row for each of `events` (default every event with a database)
    that has one; a shot the model did not label gets a row that says so."""
    from . import fdp
    from .errors import NoDataError, UnknownEventError

    out = []
    for event in fdp.events(release=release) if events is None else events:
        try:
            db = fdp.open(event, release=release)
        except (UnknownEventError, NoDataError):
            continue
        tag = MODEL_TAG
        try:
            frame = _model_runs(shot, db, taxonomy)
            notch = db.shots.loc[db.shots["shot"] == int(shot)].get("notch")
            if notch is not None and bool(notch.any()):
                tag += ", notched"  # display only: never a filter
        except NoDataError:
            frame = pd.DataFrame(columns=["t_start_ms", "t_end_ms"])
        out.append(_Row(event, MODEL_TAG, _row_label(event, tag, taxonomy), frame))
    return out


def _with_model_rows(rows: list[_Row], extra: list[_Row]) -> list[_Row]:
    """`rows` with each model row placed after the last row of its event."""
    out = list(rows)
    for row in extra:
        last = max((i for i, r in enumerate(out) if r.event == row.event), default=None)
        out.insert(len(out) if last is None else last + 1, row)
    return out


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


def _spec_list(signals) -> list:
    """`signals` as a list of specs. A tuple is one spec only when it is a
    `(group, channels)` pair as `signals.parse` takes it (channels None, an int or
    ints); `("filterscopes:1", "neutron_rate:3")` is two specs."""
    if signals is None:
        return []
    if isinstance(signals, str):
        return [signals]
    if isinstance(signals, tuple) and len(signals) == 2 and _is_pair(signals):
        return [signals]
    return list(signals)


def _is_pair(spec: tuple) -> bool:
    group, channels = spec
    if not isinstance(group, str):
        return False
    if channels is None or isinstance(channels, (int, np.integer)):
        return True
    return (
        isinstance(channels, (list, tuple, np.ndarray))
        and len(channels) > 0
        and all(isinstance(c, (int, np.integer)) for c in channels)
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
    model: bool = False,
):
    """One shot as signal panels over its label timeline; returns a matplotlib Figure.

    events: event ids to draw; None draws every event that has rows for this shot.
        A requested (event, tier) with no rows still gets a row that says so.
    signals: signal specs to stack above the labels, read from the processed corpus
        (`tokamaksi.signals.parse`: `"neutron_rate:0"`, `"bolo:0,1"`; `corpus=` or
        `$TOKAMAKSI_CORPUS` names the folder; needs `tokamaksi[h5]`).
    tier: the tier or tiers to draw; each event's rows follow the order given.
    t_range: `(start, stop)` in ms; default the labelled extent plus 2 %.
    model: also draw the all-shot 1 ms model labels (`tokamaksi.fdp`) of each event
        that has them, one row tagged `[1 ms model]` under the event's labels. They
        are model output, not labels. Needs `$TOKAMAKSI_FDP_LABELS` or a release
        that ships `fdp_1ms/`.
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
    row_list = _make_rows(labels, taxonomy, pairs, tiers)
    extra = []
    if model:
        names = [events] if isinstance(events, str) else events
        extra = _model_rows(shot, names, taxonomy, rel)
        row_list = _with_model_rows(row_list, extra)
    if not row_list:
        row_list = [
            _Row("", t, f"(no events) [{TIER_TAG.get(t, t)}]", labels) for t in tiers
        ]
    drawn = [f for f in (labels, *(row.frame for row in extra)) if not f.empty]
    window = t_range or (_extent(pd.concat(drawn)) if drawn else None)
    specs = _spec_list(signals)
    for spec in specs:
        signals_module.parse(spec)  # a malformed spec fails before any file is read
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
    handles = _draw_rows(
        axes[-1], row_list, float(window[0]), float(window[1]), taxonomy
    )
    figure.legend(handles=handles, loc="outside right upper", fontsize=7, frameon=False)
    figure.suptitle(title or f"shot {int(shot)}  (release {rel.tag})", fontsize=10)
    return figure
