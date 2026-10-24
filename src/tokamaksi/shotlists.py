"""The shot lists a release carries: gold, silver, holdout and the cohort split."""

from __future__ import annotations

import os

import pandas as pd

from .release import Release, open_release


def gold(release: str | os.PathLike | Release | None = None) -> pd.DataFrame:
    """The gold shots (`shot, set, split, run_day`): split `test`, evaluation only."""
    return open_release(release).shots("gold")


def silver(release: str | os.PathLike | Release | None = None) -> pd.DataFrame:
    """The silver review list (`shot, set, split`)."""
    return open_release(release).shots("silver")


def holdout(release: str | os.PathLike | Release | None = None) -> pd.DataFrame:
    """The roster holdouts (`shot, set, split, events, silver`): kept from training."""
    return open_release(release).shots("holdout")


def cohort_split(release: str | os.PathLike | Release | None = None) -> pd.DataFrame:
    """Every cohort shot with its split (`shot, split, gold, holdout`; split is
    train, val or test)."""
    return open_release(release).shots("cohort_split")
