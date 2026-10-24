"""A synthetic release, written from tokamaksi's own schema tables at test time.

Nothing here is real: the shots are 9000xx (DIII-D shots are below 210000), the
source table is fictional and every number is made up. The files are written
fresh into a temporary folder for each test session and never committed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tokamaksi.schema import GRID_COLUMNS, LABEL_COLUMNS, SHOT_COLUMNS

SILVER = [900001, 900002, 900003, 900004, 900005]
GOLD = [900006, 900007, 900008]
HOLDOUT = 900005
VAL = 900004

_ARROW = {
    "int32": pa.int32(),
    "int16": pa.int16(),
    "int8": pa.int8(),
    "float64": pa.float64(),
    "string": pa.large_string(),
    "boolean": pa.bool_(),
}
_CATEGORIES = {
    "edge_localized_mode": {
        0: "absent",
        1: "present",
        2: "uncertain",
        3: "not_observable",
    },
    "confinement": {
        0: "absent",
        1: "high",
        2: "low",
        3: "qh",
        4: "wpqh",
        5: "uncertain",
        6: "imode",
    },
    "mhd_n1": {0: "absent", 1: "present", 2: "uncertain", 3: "not_observable"},
    "mhd_n1.m2": {0: "absent", 1: "present", 2: "uncertain", 3: "not_observable"},
    "sawtooth_oscillation": {
        0: "absent",
        1: "present",
        2: "uncertain",
        3: "not_observable",
    },
    # As in the real taxonomy, category 4 says nobody assessed the time.
    "resistive_wall_mode": {
        0: "absent",
        1: "present",
        2: "uncertain",
        3: "not_observable",
        4: "unassessed",
    },
}


def _table(rows: list[dict], columns: dict[str, str]) -> pa.Table:
    schema = pa.schema([(name, _ARROW[kind]) for name, kind in columns.items()])
    return pa.Table.from_pylist(rows, schema=schema)


def _label(
    shot,
    event,
    category,
    start,
    stop,
    *,
    tier,
    split,
    flag=None,
    dataset=None,
    layer="saved",
    method="synthetic",
    version="",
):
    return {
        "shot": shot,
        "event": event,
        "category": category,
        "category_name": _CATEGORIES[event][category],
        "t_start_ms": float(start),
        "t_end_ms": float(stop),
        "confidence": None,
        "iscrowd": flag,
        "attrs": "" if flag is None else json.dumps({"iscrowd": int(flag)}),
        "tier": tier,
        "layer": layer,
        "dataset": dataset or f"{event}/{tier}/synthetic",
        "method": method,
        "version": version,
        "reviewer": "R1" if tier == "reviewed" else "",
        "split": split,
    }


def _labels() -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}

    def add(tier, split, *rows):
        out.setdefault((tier, split), []).extend(rows)

    elm, conf = "edge_localized_mode", "confinement"
    # One shot with a span holding three individuals, plus assessed-absent time.
    add(
        "reviewed",
        "train",
        _label(900001, elm, 0, 0, 500, tier="reviewed", split="train"),
        _label(900001, elm, 1, 500, 1500, tier="reviewed", split="train", flag=True),
        _label(900001, elm, 1, 600, 610, tier="reviewed", split="train", flag=False),
        _label(900001, elm, 1, 700, 710, tier="reviewed", split="train", flag=False),
        _label(900001, elm, 1, 800, 800, tier="reviewed", split="train", flag=False),
        _label(900001, conf, 1, 0, 1000, tier="reviewed", split="train"),
        _label(900001, conf, 2, 1000, 2000, tier="reviewed", split="train"),
        _label(
            900002, "mhd_n1", 1, 200, 900, tier="reviewed", split="train", flag=True
        ),
        _label(
            900002, "mhd_n1.m2", 1, 300, 600, tier="reviewed", split="train", flag=True
        ),
        _label(
            900003, "sawtooth_oscillation", 1, 100, 400, tier="reviewed", split="train"
        ),
    )
    add(
        "reviewed",
        "test",
        _label(900006, elm, 1, 100, 400, tier="reviewed", split="test", flag=True),
    )
    add(
        "original",
        "train",
        _label(
            900001,
            conf,
            1,
            0,
            2000,
            tier="original",
            split="train",
            layer="format",
            dataset="confinement/format/example_lab_table",
        ),
        _label(
            900002,
            conf,
            3,
            0,
            1000,
            tier="original",
            split="train",
            layer="format",
            dataset="confinement/format/example_lab_table",
        ),
    )
    add(
        "original",
        "holdout",
        _label(
            HOLDOUT,
            conf,
            2,
            0,
            800,
            tier="original",
            split="holdout",
            layer="format",
            dataset="confinement/format/example_lab_table",
        ),
    )
    add(
        "derived",
        "train",
        _label(
            900002,
            elm,
            1,
            100,
            110,
            tier="derived",
            split="train",
            flag=False,
            layer="extend",
            dataset="edge_localized_mode/extend_example_rule",
        ),
    )
    # Two datasets in one tier on one shot, and time the tables say nobody assessed.
    rwm = "resistive_wall_mode"
    a = {"dataset": f"{rwm}/extend_rule_a/windows", "method": "rule_a", "version": "v2"}
    b = {"dataset": f"{rwm}/extend_rule_b/onsets", "method": "rule_b", "version": "v1"}
    der = {"tier": "derived", "split": "train", "layer": "extend"}
    add(
        "derived",
        "train",
        _label(900003, rwm, 1, 100, 300, flag=True, **der, **a),
        _label(900003, rwm, 4, 300, 900, **der, **a),
        _label(900003, rwm, 3, 900, 1200, **der, **a),
        _label(900003, rwm, 1, 150, 160, flag=False, **der, **b),
    )
    add(
        "source",
        "train",
        _label(
            900001,
            elm,
            1,
            0,
            3000,
            tier="source",
            split="train",
            layer="source",
            dataset="edge_localized_mode/raw/example_draft",
        ),
    )
    return out


def _run(shot, event, frame, n, category, *, split):
    return {
        "shot": shot,
        "event": event,
        "frame": frame,
        "n_frames": n,
        "t_start_ms": frame * 10,
        "t_end_ms": (frame + n) * 10,
        "category": category,
        "tier": "reviewed",
        "layer": "saved",
        "dataset": f"{event}/review/labels",
        "split": split,
    }


def _grid_runs(shots: list[int], split: str) -> list[dict]:
    """Per shot: ELM runs from frame 0 with a gap of unassessed frames, and a
    confinement run that starts at frame 20 (so events start at different frames)."""
    rows = []
    for shot in shots:
        rows += [
            _run(shot, "edge_localized_mode", 0, 50, 0, split=split),
            _run(shot, "edge_localized_mode", 50, 100, 1, split=split),
            # frames 150..199 are unassessed
            _run(shot, "edge_localized_mode", 200, 30, 0, split=split),
            _run(shot, "confinement", 20, 80, 2, split=split),
        ]
    return rows


def _taxonomy() -> dict:
    events = {}
    for name, cats in _CATEGORIES.items():
        entry = {"display": name, "on_review_page": True, "categories": cats}
        if "." in name:
            entry["parent"] = name.split(".")[0]
            entry["m"], entry["n"] = 2, 1
            entry["group"] = "MHD mode"
        events[name] = entry
    events["mhd_n1"]["group"] = "MHD mode"
    events["edge_localized_mode"]["display"] = "ELM"
    events["sawtooth_oscillation"]["on_review_page"] = False
    events["sawtooth_oscillation"]["folded_into"] = {
        "event": "mhd_n1",
        "categories": {1: 1, 2: 2},
    }
    return {
        "release": "v1",
        "grid": {
            "grid_ms": 10,
            "unassessed": -1,
            "rule": "10 ms frames; -1 unassessed.",
            "zero": {name: "absent: assessed and not seen" for name in events}
            | {"confinement": "the owner saved no regime here"},
        },
        "tiers": {"reviewed": "synthetic"},
        "events": events,
        "third_party_sources": [
            {"stem": "example_lab_table", "provenance": "Example Lab table"}
        ],
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_release(path: Path, *, schema: int | None = None, tag: str = "v1") -> Path:
    """Write a complete synthetic release into `path` and return it."""
    path = Path(path)
    for folder in ("labels", "grid", "shots"):
        (path / folder).mkdir(parents=True, exist_ok=True)
    counts = {}
    for (tier, split), rows in _labels().items():
        (path / "labels" / tier).mkdir(exist_ok=True)
        pq.write_table(
            _table(rows, LABEL_COLUMNS), path / "labels" / tier / f"{split}.parquet"
        )
        frame = pd.DataFrame(rows)
        counts[f"{tier}/{split}"] = {
            "rows": len(rows),
            "shots": int(frame.shot.nunique()),
            "events": int(frame.event.nunique()),
        }

    pq.write_table(
        _table(_grid_runs(SILVER, "train"), GRID_COLUMNS),
        path / "grid" / "silver.parquet",
    )
    pq.write_table(
        _table(_grid_runs(GOLD, "test"), GRID_COLUMNS), path / "grid" / "gold.parquet"
    )

    lists = {
        "gold": [
            {"shot": s, "set": "gold", "split": "test", "run_day": 20260101}
            for s in GOLD
        ],
        "silver": [{"shot": s, "set": "silver", "split": "train"} for s in SILVER],
        "holdout": [
            {
                "shot": HOLDOUT,
                "set": "holdout",
                "split": "holdout",
                "events": "edge_localized_mode",
                "silver": True,
            }
        ],
        "cohort_split": [
            {
                "shot": s,
                "split": "val" if s == VAL else "train",
                "gold": False,
                "holdout": s == HOLDOUT,
            }
            for s in SILVER
        ]
        + [{"shot": s, "split": "test", "gold": True, "holdout": False} for s in GOLD],
    }
    for name, rows in lists.items():
        pq.write_table(
            _table(rows, SHOT_COLUMNS[name]), path / "shots" / f"{name}.parquet"
        )

    datasets = pd.DataFrame(
        [
            {
                "dataset": "edge_localized_mode/review/labels",
                "event": "edge_localized_mode",
                "tier": "reviewed",
                "layer": "saved",
                "method": "review",
                "version": "",
                "made_at": "",
                "made_by": "",
                "n_rows": 5,
                "n_shots": 1,
                "categories": "{}",
                "shipped": True,
                "why": "",
                "grid_frames": 0,
            },
            {
                "dataset": "confinement/format/example_lab_table",
                "event": "confinement",
                "tier": "original",
                "layer": "format",
                "method": "format",
                "version": "2026_v1",
                "made_at": "",
                "made_by": "",
                "n_rows": 3,
                "n_shots": 3,
                "categories": "{}",
                "shipped": True,
                "why": "",
                "grid_frames": 0,
            },
        ]
    )
    datasets.to_parquet(path / "datasets.parquet")
    (path / "taxonomy.yaml").write_text(
        yaml.safe_dump(_taxonomy(), sort_keys=False), encoding="utf-8"
    )
    (path / "signals.yaml").write_text(
        yaml.safe_dump(
            {"edge_localized_mode": {"model": "example", "task": "detector"}}
        ),
        encoding="utf-8",
    )
    (path / "README.md").write_text("# Synthetic release\n", encoding="utf-8")
    (path / "LICENSE.md").write_text("Synthetic test data.\n", encoding="utf-8")

    listed = {
        p.relative_to(path).as_posix(): {"bytes": p.stat().st_size, "sha256": _sha(p)}
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "release.json" and not p.name.startswith(".")
    }
    manifest = {
        "release": tag,
        "repo": "example/synthetic",
        "made_at": "2026-01-01T00:00:00+00:00",
        "labels": counts,
        "grids": {"silver": {"shots": len(SILVER)}, "gold": {"shots": len(GOLD)}},
        "files": listed,
    }
    if schema is not None:
        manifest["schema"] = schema
    (path / "release.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def make_corpus(folder: Path, shot: int = 900001) -> Path:
    """A synthetic signal file: four live groups (one with dead channels) and a sentinel."""
    import h5py
    import numpy as np

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{shot}_processed.h5"
    with h5py.File(path, "w") as handle:
        for name, rows, rate in (
            ("neutron_rate", 2, 1000.0),
            ("bolo", 3, 5000.0),
            ("ece", 2, 50000.0),
            ("filterscopes", 4, 2000.0),
        ):
            n = int(3 * rate)
            t = (np.arange(n) / rate).astype("float32")
            y = np.vstack(
                [np.sin(2 * np.pi * (i + 1) * t) + i for i in range(rows)]
            ).astype("float32")
            if name == "filterscopes":
                y[:2] = np.nan  # channels that recorded nothing are NaN throughout
            group = handle.create_group(name)
            group["xdata"] = t
            group["ydata"] = y
        sentinel = handle.create_group("pinj")
        sentinel["xdata"] = np.zeros(1, dtype="float32")
        sentinel["ydata"] = np.zeros((8, 1), dtype="float32")
    return path
