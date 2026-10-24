"""What a release holds: tiers, splits, columns and the grid rule.

The names and texts below are copied from FusionAIHub's `labeler.events.hf_export`
(`TIERS`, `TIER_TEXT`, `SPLITS`) and `labeler.events.grid` (`GRID_MS`, `UNASSESSED`)
at commit 52c2a2bd. tokamaksi imports nothing from FusionAIHub, so a change there
is a change here by hand; `tests/test_schema.py` pins the values.
"""

from __future__ import annotations

from collections.abc import Iterable

from .errors import SchemaError

#: The newest release schema this tokamaksi reads. `release.json` has no `schema`
#: key in v1, which reads as 1.
RELEASE_SCHEMA_MAX = 1

#: Frame length of the 10 ms grid (FusionAIHub `labeler.events.grid.GRID_MS`).
GRID_MS = 10
#: Grid category of a frame nobody assessed (`labeler.events.grid.UNASSESSED`).
#: It is not zero: zero is "assessed, not seen" where the taxonomy says so.
UNASSESSED = -1

TIER_ORDER = ("reviewed", "original", "derived", "source")
TIER_TEXT = {
    "reviewed": "saved by a reviewer on the review page (one label per shot and event)",
    "original": "third-party label tables, reformatted to the interval schema without "
    "changing what they say",
    "derived": "tables our producers made (a rule or a model each)",
    "source": "machine drafts: nobody reviewed them, so they are not labels",
}
#: `test` is the gold shots (evaluation only), `holdout` the shots an event roster
#: holds out of training and tuning, `train` the rest.
SPLITS = ("train", "holdout", "test")

#: The columns of `labels/<tier>/<split>.parquet` and their dtypes in the writer.
LABEL_COLUMNS = {
    "shot": "int32",
    "event": "string",
    "category": "int16",
    "category_name": "string",
    "t_start_ms": "float64",
    "t_end_ms": "float64",
    "confidence": "float64",
    "iscrowd": "boolean",
    "attrs": "string",
    "tier": "string",
    "layer": "string",
    "dataset": "string",
    "method": "string",
    "version": "string",
    "reviewer": "string",
    "split": "string",
}
#: The columns of `grid/<name>.parquet`: one row per run of equal frames.
GRID_COLUMNS = {
    "shot": "int32",
    "event": "string",
    "frame": "int32",
    "n_frames": "int32",
    "t_start_ms": "int32",
    "t_end_ms": "int32",
    "category": "int8",
    "tier": "string",
    "layer": "string",
    "dataset": "string",
    "split": "string",
}
#: The shot lists a release carries, and their columns.
SHOT_COLUMNS = {
    "gold": {"shot": "int32", "set": "string", "split": "string", "run_day": "int32"},
    "silver": {"shot": "int32", "set": "string", "split": "string"},
    "holdout": {
        "shot": "int32",
        "set": "string",
        "split": "string",
        "events": "string",
        "silver": "boolean",
    },
    "cohort_split": {
        "shot": "int32",
        "split": "string",
        "gold": "boolean",
        "holdout": "boolean",
    },
}
#: What `open_release` needs on disk; `tables/` is never fetched or read.
REQUIRED_FILES = ("release.json", "taxonomy.yaml", "datasets.parquet")
REQUIRED_DIRS = ("labels", "grid", "shots")
#: The Hub files `pull` fetches (everything but `tables/`, the editable mirror).
PULL_PATTERNS = (
    "release.json",
    "taxonomy.yaml",
    "signals.yaml",
    "datasets.parquet",
    "README.md",
    "LICENSE.md",
    "labels/**",
    "grid/**",
    "shots/**",
)


def require_columns(have: Iterable[str], need: Iterable[str], what: str) -> None:
    """Raise `SchemaError` naming the columns of `what` that `have` lacks."""
    missing = [name for name in need if name not in set(have)]
    if missing:
        raise SchemaError(
            f"{what} lacks the column(s) {', '.join(missing)}: the release was "
            f"written by a different schema than this tokamaksi reads"
        )
