import pytest
from release_factory import GOLD, HOLDOUT, SILVER, VAL

import tokamaksi as tk


def test_the_four_lists(rel):
    gold = tk.shotlists.gold(rel)
    assert list(gold.columns) == ["shot", "set", "split", "run_day"]
    assert list(gold.shot) == GOLD and set(gold.split) == {"test"}
    assert list(tk.shotlists.silver(rel).shot) == SILVER
    holdout = tk.shotlists.holdout(rel)
    assert list(holdout.columns) == ["shot", "set", "split", "events", "silver"]
    assert list(holdout.shot) == [HOLDOUT]
    cohort = tk.shotlists.cohort_split(rel)
    assert set(cohort.split) == {"train", "val", "test"}
    assert cohort.set_index("shot").loc[VAL, "split"] == "val"


def test_a_list_the_release_lacks(tmp_path, synthetic):
    import shutil

    copy = tmp_path / "c"
    shutil.copytree(synthetic, copy)
    (copy / "shots" / "holdout.parquet").unlink()
    with pytest.raises(tk.MissingPartError, match="no holdout shot list"):
        tk.shotlists.holdout(copy)
